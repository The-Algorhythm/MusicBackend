import os
from django.http import JsonResponse
from django.http import HttpResponseRedirect
import json
import torch


def test_torch(request):
    return JsonResponse({"cuda_available": torch.cuda.is_available(), "rand_tensor": torch.rand(5, 3).numpy().tolist()})
