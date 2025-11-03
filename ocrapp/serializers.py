from rest_framework import serializers
from .models import ArticleInfo

class ArticleInfoSerializer(serializers.ModelSerializer):
    class Meta:
        model = ArticleInfo
        fields = "__all__"