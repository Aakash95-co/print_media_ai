from django.db import models

class ArticleInfo(models.Model):
    pdf_name = models.CharField(max_length=255)
    page_number = models.IntegerField(null=True)
    article_id = models.CharField(max_length=50)
    gujarati_text = models.TextField(blank=True)
    gujarati_title = models.TextField(blank=True,)
    translated_text = models.TextField(blank=True, null=True)
    sentiment = models.CharField(max_length=100)
    district = models.CharField(max_length=100)
    dist_token = models.CharField(max_length=100, blank=True, null=True)
    article_type = models.CharField(max_length=100)
    article_category = models.CharField(max_length=100)
    image = models.ImageField(upload_to="articles/")
    created_at = models.DateTimeField(auto_now_add=True)
    # phase - 2
    is_govt = models.BooleanField(default=False)
    govt_word = models.CharField(max_length=255, blank=True, null=True)
    govt_word_rule_based = models.CharField(max_length=255, blank=True, null=True)
    category_word = models.CharField(max_length=100, blank=True) 
    distict_word =  models.CharField(max_length=100, blank=True, null=True)
    prabhag =  models.CharField(max_length=255, blank=True, null=True) 
    prabhag_ID = models.IntegerField(null=True)
    Dcode = models.IntegerField(null=True)
    Tcode = models.IntegerField(null=True)
    cat_Id = models.IntegerField(null=True)
    sentiment_gravity = models.FloatField(null=True)
    is_govt_push_nic  = models.BooleanField(default=False)
    pdf_link = models.CharField(max_length=500, blank=True, null=True)
    remarks = models.TextField(blank=True, null=True)
    
    def __str__(self):
        return f"{self.pdf_name} - {self.article_id}"
