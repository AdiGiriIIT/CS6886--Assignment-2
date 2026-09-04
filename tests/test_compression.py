import math
import unittest
import torch
from torch import nn
from src.compression import fold_conv_bn, pack_signed, unpack_signed, weight_size_breakdown
from src.qat import precision_for_epoch
from src.quantization import QuantizedResidualAdd, fake_quantize, integer_range, lsq_scale_init

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

    def test_residual_add_uses_one_signed_scale_and_transition(self):
        add = QuantizedResidualAdd(4)
        result = add(torch.tensor([-1.0]), torch.tensor([2.0]))
        self.assertEqual(tuple(result.shape), (1,))
        self.assertEqual(len(list(add.parameters())), 1)
        self.assertEqual(precision_for_epoch(1, 4, 4, 1), (8, 8))
        self.assertEqual(precision_for_epoch(2, 4, 4, 1), (6, 6))
        self.assertEqual(precision_for_epoch(3, 4, 4, 1), (4, 4))

if __name__ == "__main__": unittest.main()
