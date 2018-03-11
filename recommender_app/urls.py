from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, {'user_id': None}),
    path('<str:user_id>', views.index),
    path('performance/', views.performance),
    path('tech-stack/', views.tech_stack),
    path('amazon-detail/<str:asin>', views.amazon_detail),
]
