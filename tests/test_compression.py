import math
import unittest
import torch
from torch import nn
from src.compression import (QuantizedInvertedResidual, activation_liveness, build_quantized_model,
                             export_packed_model, export_sparse_packed_model, fold_batch_norms, fold_conv_bn, pack_bitmap,
                             pack_signed, unpack_bitmap, unpack_signed, weight_size_breakdown)
from src.pruning import enforce_masks, global_magnitude_masks, masked_optimizer_step, validate_masks
from src.qat import precision_for_epoch
from src.prune_qat import scheduled_sparsity
from src.distill import initialize_from_teacher
from src.quantization import (ActivationFakeQuantizer, QuantizedResidualAdd,
                              apply_mixed_weight_policy, fake_quantize, integer_range, lsq_scale_init,
                              set_quantizer_bits)

class CompressionTests(unittest.TestCase):
    def test_ranges_and_levels(self):
        self.assertEqual(integer_range(4, True), (-7, 7)); self.assertEqual(integer_range(4, False), (0, 15))
        values = torch.tensor([-100., -7., -.49, .49, 7., 100.])
        self.assertTrue(torch.equal(fake_quantize(values, torch.tensor(1.), 4, True), torch.tensor([-7., -7., 0., 0., 7., 7.])))
    def test_scale_positive_and_per_channel(self):
        scale = lsq_scale_init(torch.tensor([[[[0.]]], [[[2.]]]]), 4, True, per_channel=True)
        self.assertEqual(tuple(scale.shape), (2, 1, 1, 1)); self.assertTrue(bool((scale > 0).all()))
    def test_ste_and_clipping_gradient(self):
        value = torch.tensor([-9., 2.], requires_grad=True); fake_quantize(value, torch.tensor(1.), 4, True).sum().backward()
        self.assertTrue(torch.equal(value.grad, torch.tensor([0., 1.])))
    def test_pack_round_trip_and_padding(self):
        for bits, values in ((3, torch.tensor([-3, -1, 0, 3, 2])), (4, torch.tensor([-7, 7, 0]))):
            payload = pack_signed(values, bits); self.assertEqual(len(payload), math.ceil(values.numel() * bits / 8))
            self.assertTrue(torch.equal(unpack_signed(payload, values.numel(), bits), values))
    def test_fold_equivalence_and_accounting(self):
        torch.manual_seed(1); conv, bn = nn.Conv2d(3, 4, 3, bias=False), nn.BatchNorm2d(4)
        conv.eval(); bn.eval(); x = torch.randn(2, 3, 8, 8); self.assertTrue(torch.allclose(bn(conv(x)), fold_conv_bn(conv, bn)(x), atol=1e-5))
        size = weight_size_breakdown(conv, 4); self.assertEqual(size.packed_weight_bytes, math.ceil(conv.weight.numel() / 2)); self.assertGreater(size.compressed_bytes, size.packed_weight_bytes)

    def test_packed_artifact_matches_accounting_and_folds_bn(self):
        torch.manual_seed(2)
        base = nn.Sequential(nn.Conv2d(3, 4, 3, bias=False), nn.BatchNorm2d(4), nn.ReLU6(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 2)).eval()
        quant = build_quantized_model(base, 3, 4).eval()
        x = torch.randn(1, 3, 8, 8); quant(x)  # initialize any lazy activation scales
        folded = fold_batch_norms(quant)
        self.assertFalse(any(isinstance(m, nn.BatchNorm2d) for m in folded.modules()))
        with __import__("tempfile").TemporaryDirectory() as directory:
            path = __import__("pathlib").Path(directory) / "model.qpk"
            size = export_packed_model(quant, path)
            self.assertEqual(path.stat().st_size, size.total_bytes)
            self.assertEqual(size.packed_weight_bytes, sum(math.ceil(m.weight.numel() * 3 / 8) for m in quant.modules() if hasattr(m, "weight_quantizer")))
        memory = activation_liveness(quant, x)
        self.assertGreater(memory.fp32_peak_live_bytes, memory.quantized_peak_live_bytes)
        self.assertGreater(memory.event_count, 0)

    def test_residual_add_uses_one_signed_scale_and_transition(self):
        add = QuantizedResidualAdd(4)
        result = add(torch.tensor([-1.0]), torch.tensor([2.0]))
        self.assertEqual(tuple(result.shape), (1,))
        self.assertEqual(len(list(add.parameters())), 1)
        self.assertEqual(precision_for_epoch(1, 4, 4, 1), (8, 8))
        self.assertEqual(precision_for_epoch(2, 4, 4, 1), (6, 6))
        self.assertEqual(precision_for_epoch(3, 4, 4, 1), (4, 4))

    def test_non_residual_projection_has_signed_output_quantizer(self):
        block = nn.Module()
        block.conv = nn.Identity()
        block.use_res_connect = False
        quantized = QuantizedInvertedResidual(block, 4)
        self.assertIsNone(quantized.residual_add)
        self.assertIsNotNone(quantized.output_quantizer)
        self.assertTrue(quantized.output_quantizer.signed)
        self.assertEqual(quantized.output_quantizer.bits, 4)
        self.assertFalse(torch.equal(quantized(torch.tensor([1.0])), torch.tensor([1.0])))

    def test_precision_transition_rescales_steps_and_preserves_uninitialized_activation(self):
        signed = ActivationFakeQuantizer(8, signed=True, initial_scale=1.0)
        relu6 = ActivationFakeQuantizer(8, signed=False, initial_scale=6 / 255)
        pending = ActivationFakeQuantizer(8, signed=True)
        wrapper = nn.Module()
        wrapper.signed, wrapper.relu6, wrapper.pending = signed, relu6, pending
        set_quantizer_bits(wrapper, 4, 4)
        self.assertAlmostEqual(float(signed.scale), math.sqrt(127 / 7), places=6)
        self.assertAlmostEqual(float(relu6.scale), 6 / 15, places=6)
        self.assertAlmostEqual(float(pending.scale), 1.0, places=6)
        self.assertEqual(pending.bits, 4)

    def test_mixed_weight_policy(self):
        base = nn.Sequential(nn.Conv2d(3, 4, 3), nn.ReLU6(),
                             nn.Conv2d(4, 4, 3, groups=4), nn.ReLU6(),
                             nn.Flatten(), nn.Linear(64, 2)).eval()
        quant = build_quantized_model(base, 4, 6)
        realized = apply_mixed_weight_policy(quant, 4, depthwise_bits=6,
                                             first_last_bits=8)
        self.assertEqual(list(realized.values()), [8, 6, 8])

    def test_bitmap_and_sparse_export_are_byte_exact(self):
        bitmap = pack_bitmap(torch.tensor([True, False, True, True, False, False, False, True, True]))
        self.assertTrue(torch.equal(unpack_bitmap(bitmap, 9), torch.tensor([True, False, True, True, False, False, False, True, True])))
        # The first convolution is 1x1 W4 and therefore the only eligible sparse tensor.
        base = nn.Sequential(nn.Conv2d(3, 4, 1, bias=False), nn.BatchNorm2d(4), nn.ReLU6(), nn.Flatten(), nn.Linear(4 * 8 * 8, 2)).eval()
        quant = build_quantized_model(base, 4, 6).eval(); quant(torch.randn(1, 3, 8, 8))
        masks, expected = global_magnitude_masks(quant, .5); enforce_masks(quant, masks)
        summary = validate_masks(quant, masks)
        self.assertEqual(summary.masked_values, expected.masked_values)
        with __import__("tempfile").TemporaryDirectory() as directory:
            path = __import__("pathlib").Path(directory) / "sparse.qpk"
            size = export_sparse_packed_model(quant, path, masks)
            self.assertEqual(path.stat().st_size, size.total_bytes)
            self.assertEqual(size.masked_weight_values, expected.masked_values)
            self.assertGreater(size.bitmap_bytes, 0)

    def test_masked_optimizer_step_prevents_minibatch_regrowth(self):
        base = nn.Sequential(nn.Conv2d(3, 4, 1, bias=False), nn.ReLU6()).eval()
        quant = build_quantized_model(base, 4, 6)
        masks, _ = global_magnitude_masks(quant, .5); enforce_masks(quant, masks)
        optimizer = torch.optim.SGD(quant.parameters(), lr=.1, momentum=.9)
        loss = quant(torch.randn(2, 3, 4, 4)).sum(); loss.backward()
        masked_optimizer_step(optimizer, quant, masks)
        validate_masks(quant, masks)

    def test_gradual_sparsity_reaches_target_monotonically(self):
        values = [scheduled_sparsity(.5, .3, epoch, 4) for epoch in range(1, 7)]
        self.assertEqual(values[-1], .5)
        self.assertEqual(values[-2], .5)
        self.assertTrue(all(left <= right for left, right in zip(values, values[1:])))

    def test_narrow_student_gets_teacher_initialization(self):
        teacher, student = nn.Sequential(nn.Linear(4, 3)), nn.Sequential(nn.Linear(2, 2))
        with torch.no_grad():
            teacher[0].weight.copy_(torch.arange(12).reshape(3, 4))
            teacher[0].bias.copy_(torch.arange(3))
        summary = initialize_from_teacher(student, teacher)
        self.assertEqual(summary["copied_values"], summary["student_values"])
        self.assertTrue(torch.equal(student[0].weight, teacher[0].weight[:2, :2]))
        self.assertTrue(torch.equal(student[0].bias, teacher[0].bias[:2]))

if __name__ == "__main__": unittest.main()
