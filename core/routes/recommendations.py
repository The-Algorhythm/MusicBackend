import os
from django.http import JsonResponse
from django.http import HttpResponseRedirect
import json
import torch
import time

from ..spotify.canvas.canvas import get_canvases


def test_torch(request):
    return JsonResponse({"cuda_available": torch.cuda.is_available(), "rand_tensor": torch.rand(5, 3).numpy().tolist()})


def get_recommendations(request):
    start = time.time()
    canvases = get_canvases(["spotify:track:0baNzeUcPQnQSagpe8T0mD", "spotify:track:6B0fJdJscs6PV9IhoVPIw9"])
    return JsonResponse({"canvases": canvases, "time": time.time()-start})
