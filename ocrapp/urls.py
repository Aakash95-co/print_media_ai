#from django.urls import path
#from .views import OCRUploadView, ArticleListView

#urlpatterns = [
#    path("upload/", OCRUploadView.as_view(), name="upload_pdf"),
#    path("articles/", ArticleListView.as_view(), name="list_articles"),
#]


from django.urls import path
from rest_framework.authtoken.views import obtain_auth_token
from .views import ocr_upload_view, ArticleListView

urlpatterns = [
    path("login/", obtain_auth_token, name="login"),
    path("upload/", ocr_upload_view, name="upload_pdf"),
    path("articles/", ArticleListView.as_view(), name="article_list"),
]
