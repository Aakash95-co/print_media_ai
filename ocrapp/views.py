from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import ArticleInfo
from .serializers import ArticleInfoSerializer
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view
from django.conf import settings
import os
from datetime import datetime
from .tasks import process_pdf_task  # <--- IMPORT THE TASK, NOT THE UTIL

from django.contrib.auth.models import User
from django.contrib.auth import authenticate
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework.authtoken.models import Token
from rest_framework import status
from django.utils import timezone
from datetime import timedelta


# --- 1. REGISTER (Direct Insert) ---
@api_view(['POST'])
@permission_classes([AllowAny])  # Open endpoint to create users
def register_view(request):
    username = request.data.get('username')
    password = request.data.get('password')
    email = request.data.get('email', '')

    if not username or not password:
        return Response({"error": "Username and Password required"}, status=status.HTTP_400_BAD_REQUEST)

    if User.objects.filter(username=username).exists():
        return Response({"error": "User already exists"}, status=status.HTTP_400_BAD_REQUEST)

    # Create the user
    user = User.objects.create_user(username=username, password=password, email=email)

    return Response({"message": f"User {username} created successfully"}, status=status.HTTP_201_CREATED)


# --- 2. LOGIN (Get 15-Day Token) ---
@api_view(['POST'])
@permission_classes([AllowAny])
def login_view(request):
    username = request.data.get('username')
    password = request.data.get('password')

    user = authenticate(username=username, password=password)

    if not user:
        return Response({"error": "Invalid Credentials"}, status=status.HTTP_400_BAD_REQUEST)

    # Get or Create Token
    token, created = Token.objects.get_or_create(user=user)

    # If token exists but is older than 15 days, refresh it now
    if not created:
        time_elapsed = timezone.now() - token.created
        if time_elapsed > timedelta(days=15):
            token.delete()
            token = Token.objects.create(user=user)

    return Response({
        "token": token.key,
        "valid_until": token.created + timedelta(days=15),
        "message": "Token is valid for 15 days"
    })




@csrf_exempt
@api_view(['POST'])
@permission_classes([IsAuthenticated]) # <--- THIS WAS MISSING
def ocr_upload_view(request):
    if request.method == 'POST':
        file = request.FILES.get("file")
        news_paper = request.data.get("news_paper", "") # news_paper
        lang = request.data.get("lang", "gu")
        article_param = request.data.get("article_param", "false")
        district_param = request.data.get("district_param", None)
        is_connect = request.data.get("is_connect", False)
        is_urgent = request.data.get("is_urgent", False)
        uuid = request.data.get("uuid", False)
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
        
        # 4. Process Asynchronously
        is_article = False
        if article_param == "article":
            is_article = True

        if int(is_urgent) == 1:
            is_urgent = True
        else:
             is_urgent = False
                       
        print(f"Queuing task for: {new_filename}")
        
        # CALL THE TASK WITH .delay()
        process_pdf_task.delay(
            pdf_path=full_path,
            news_paper=news_paper,
            pdf_link=pdf_link,
            lang="gu",
            is_article=is_article,
            article_district=district_param,
            is_connect=is_connect, 
            is_urgent=is_urgent,
            uuid=uuid
        )
        
        # Return immediate response
        return Response({"message": "File queued for processing", "pdf_link": pdf_link}, status=status.HTTP_200_OK)
    
    return Response({"error": "Method not allowed"}, status=status.HTTP_405_METHOD_NOT_ALLOWED)


from rest_framework.views import APIView
from rest_framework.response import Response
from django.utils.dateparse import parse_date

class ArticleListView(APIView):
    permission_classes = [IsAuthenticated]  # <--- THIS WAS MISSING
    def get(self, request):
        qs = ArticleInfo.objects.all().order_by("-created_at")

        # Query Params
        from_date = request.GET.get("from_date")   # yyyy-mm-dd
        to_date = request.GET.get("to_date")       # yyyy-mm-dd
        district = request.GET.get("district")
        prabhag = request.GET.get("prabhag")
        article_category = request.GET.get("article_category")
        sentiment = request.GET.get("sentiment")

        # Date Range Filter (using created_at)
        if from_date:
            qs = qs.filter(created_at__date__gte=parse_date(from_date))

        if to_date:
            qs = qs.filter(created_at__date__lte=parse_date(to_date))

        # Other Filters
        if district:
            qs = qs.filter(district__iexact=district)

        if prabhag:
            qs = qs.filter(prabhag__iexact=prabhag)

        if article_category:
            qs = qs.filter(article_category__iexact=article_category)

        if sentiment:
            qs = qs.filter(sentiment__iexact=sentiment)

        # Serialize
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
