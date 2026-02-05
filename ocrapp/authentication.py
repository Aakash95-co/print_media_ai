# authentication.py
from rest_framework.authentication import TokenAuthentication
from rest_framework import exceptions
from django.utils import timezone
from datetime import timedelta

class ExpiringTokenAuthentication(TokenAuthentication):
    def authenticate_credentials(self, key):
        model = self.get_model()
        try:
            token = model.objects.select_related('user').get(key=key)
        except model.DoesNotExist:
            raise exceptions.AuthenticationFailed('Invalid token.')

        if not token.user.is_active:
            raise exceptions.AuthenticationFailed('User inactive or deleted.')

        # CHECK IF 15 DAYS PASSED
        time_elapsed = timezone.now() - token.created
        if time_elapsed > timedelta(days=15):
            token.delete() # Delete expired token
            raise exceptions.AuthenticationFailed('Token expired (15 days). Please login again.')

        return (token.user, token)