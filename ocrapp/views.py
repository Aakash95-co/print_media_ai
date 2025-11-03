from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .utils.ocr_utils import process_pdf
from .models import ArticleInfo
from .serializers import ArticleInfoSerializer
from django.views.decorators.csrf import csrf_exempt
#from django.views.decorators.csrf import csrf_exempt

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from .utils.ocr_utils import process_pdf
from .models import ArticleInfo
from .serializers import ArticleInfoSerializer
import os
import shutil


from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
#from rest_framework.decorators import api_view, csrf_exempt
from .utils.ocr_utils import process_pdf
from .models import ArticleInfo
from .serializers import ArticleInfoSerializer

from rest_framework.decorators import api_view
from django.views.decorators.csrf import csrf_exempt


#@csrf_exempt
@csrf_exempt
@api_view(['POST'])
def ocr_upload_view(request):
    if request.method == 'POST':
        file = request.FILES.get("pdf")
        if not file:
            return Response({"error": "No PDF uploaded"}, status=status.HTTP_400_BAD_REQUEST)
        tmp_path = f"/tmp/{file.name}"
        with open(tmp_path, "wb") as f:
            for chunk in file.chunks():
                f.write(chunk)
        process_pdf(tmp_path)
        return Response({"message": "Processing complete"}, status=status.HTTP_200_OK)
    return Response({"error": "Method not allowed"}, status=status.HTTP_405_METHOD_NOT_ALLOWED)


class ArticleListView(APIView):
    def get(self, request):
        qs = ArticleInfo.objects.all().order_by("-created_at")
        serializer = ArticleInfoSerializer(qs, many=True)
        return Response(serializer.data)


#@csrf_exempt
#@method_decorator(csrf_exempt, name='dispatch')
class OCRUploadView33(APIView):
    def post(self, request):
        file = request.FILES.get("pdf")
        if not file:
            return Response({"error": "No PDF uploaded"}, status=status.HTTP_400_BAD_REQUEST)

        tmp_path = f"/tmp/{file.name}"
        with open(tmp_path, "wb") as f:
            for chunk in file.chunks():
                f.write(chunk)

        process_pdf(tmp_path)
        return Response({"message": "Processing complete"}, status=status.HTTP_200_OK)


#@csrf_exempt
#@method_decorator33(csrf_exempt, name='dispatch')
class ArticleListView33(APIView):
    def get(self, request):
        qs = ArticleInfo.objects.all().order_by("-created_at")
        serializer = ArticleInfoSerializer(qs, many=True)
        return Response(serializer.data)
