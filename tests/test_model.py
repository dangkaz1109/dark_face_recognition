import unittest
import torch
from src.model import FacenetBaseline
from src.loss import StandardTripletLoss

class TestFacenetBaseline(unittest.TestCase):
    def test_model_shapes(self):
        model = FacenetBaseline()
        x = torch.randn(2, 3, 160, 160)
        emb = model(x)
        self.assertEqual(emb.shape, (2, 512))

    def test_model_gradients(self):
        model = FacenetBaseline()
        # Ensure parameters are trainable by default
        for param in model.parameters():
            self.assertTrue(param.requires_grad)

    def test_standard_triplet_loss(self):
        criterion = StandardTripletLoss(margin=1.0)
        
        anchor = torch.tensor([[1.0, 0.0], [2.0, 2.0]], dtype=torch.float32, requires_grad=True)
        positive = torch.tensor([[1.1, 0.1], [2.1, 2.1]], dtype=torch.float32, requires_grad=True)
        negative = torch.tensor([[5.0, -4.0], [10.0, 10.0]], dtype=torch.float32, requires_grad=True)
        
        loss = criterion(anchor, positive, negative)
        self.assertGreaterEqual(loss.item(), 0.0)
        
        # Test backward pass
        loss.backward()
        self.assertIsNotNone(anchor.grad)
        self.assertIsNotNone(positive.grad)
        self.assertIsNotNone(negative.grad)

if __name__ == '__main__':
    unittest.main()
