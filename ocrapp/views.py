from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .utils.ocr_utils import process_pdf
from .models import ArticleInfo
from .serializers import ArticleInfoSerializer
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view
from django.conf import settings
import os
from datetime import datetime


@csrf_exempt
@api_view(['POST'])
def ocr_upload_view(request):
    if request.method == 'POST':
        file = request.FILES.get("file")
        news_paper = request.data.get("news_paper", "")
        lang = request.data.get("lang", "gu")
        article_param = request.data.get("article", "false")
        district_param = request.data.get("district", None)
        is_connect = request.data.get("district", False)
        if not file:
            return Response({"error": "No PDF uploaded"}, status=status.HTTP_400_BAD_REQUEST)
        
        # 1. Generate Filename: name + dd+mm+yy+min+sec+microsecond
        now = datetime.now()
        timestamp = now.strftime("%d%m%y%M%S%f")
        
        # Sanitize newspaper name
        safe_name = "".join(c for c in news_paper if c.isalnum() or c in " _-").strip().replace(" ", "_")
        if not safe_name:
            safe_name = "doc"
            
        ext = os.path.splitext(file.name)[1]
        new_filename = f"{safe_name}_{timestamp}{ext}"
        
        # 2. Save to MEDIA_ROOT/pdfs
        save_dir = os.path.join(settings.MEDIA_ROOT, "pdfs")
        os.makedirs(save_dir, exist_ok=True)
        
        full_path = os.path.join(save_dir, new_filename)
        
        with open(full_path, "wb") as f:
            for chunk in file.chunks():
                f.write(chunk)
        
        # 3. Create relative link (e.g., "pdfs/newspaper_123456.pdf")
        pdf_link = f"pdfs/{new_filename}"
        
        # 4. Process with new args
        process_pdf(full_path, news_paper, pdf_link, is_connect)
        
        return Response({"message": "Processing complete", "pdf_link": pdf_link}, status=status.HTTP_200_OK)
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
