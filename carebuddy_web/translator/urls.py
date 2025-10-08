from django.urls import path
from . import views

urlpatterns = [
    path("api/test/", views.test_api, name="test_api"),
    path("", views.home, name="home"),
]
