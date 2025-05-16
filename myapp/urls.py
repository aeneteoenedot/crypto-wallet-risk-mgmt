from django.urls import path
from .views import crypto_list
from .views import handle_form_submission

urlpatterns = [
    path('', crypto_list, name='crypto_list'),
    path('handle_form_submission/', handle_form_submission, name='handle_form_submission'),
]
