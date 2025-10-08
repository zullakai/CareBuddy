# Create your views here.
from django.http import JsonResponse
from django.shortcuts import render

# API endpoint
def test_api(request):
    return JsonResponse({"message": "Hello, this is CareBuddy API responding!"})

# Frontend page
def home(request):
    return render(request, "index.html")
