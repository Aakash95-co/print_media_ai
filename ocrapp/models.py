from django.db import models

class ArticleInfo(models.Model):
    pdf_name = models.CharField(max_length=255)
    page_number = models.IntegerField()
    article_id = models.CharField(max_length=50)
    gujarati_text = models.TextField()
    translated_text = models.TextField(blank=True, null=True)
    sentiment = models.CharField(max_length=100)
    district = models.CharField(max_length=100)
    dist_token = models.CharField(max_length=100, blank=True, null=True)
    article_type = models.CharField(max_length=100)
    article_category = models.CharField(max_length=100)
    image = models.ImageField(upload_to="articles/")
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.pdf_name} - {self.article_id}"
