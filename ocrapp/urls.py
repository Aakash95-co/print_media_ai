#from django.urls import path
#from .views import OCRUploadView, ArticleListView

#urlpatterns = [
#    path("upload/", OCRUploadView.as_view(), name="upload_pdf"),
#    path("articles/", ArticleListView.as_view(), name="list_articles"),
#]


from django.urls import path
from .views import ocr_upload_view, ArticleListView, register_view, login_view

urlpatterns = [
    path('register/', register_view, name='register'),
    path('login/', login_view, name='login'),
    path("upload/", ocr_upload_view, name="upload_pdf"),
    path("articles/", ArticleListView.as_view(), name="article_list"),
]
