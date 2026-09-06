import math
import unittest
import torch
from torch import nn
from src.compression import (QuantizedInvertedResidual, activation_liveness, build_quantized_model,
                             export_packed_model, fold_batch_norms, fold_conv_bn, pack_signed,
                             unpack_signed, weight_size_breakdown)
from src.qat import precision_for_epoch
from src.quantization import (ActivationFakeQuantizer, QuantizedResidualAdd,
                              fake_quantize, integer_range, lsq_scale_init,
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

if __name__ == "__main__": unittest.main()
