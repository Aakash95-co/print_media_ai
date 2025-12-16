from celery import shared_task
from .utils.ocr_utils import process_pdf

@shared_task(bind=True)
def process_pdf_task(self, pdf_path, news_paper, pdf_link, lang, is_article, article_district, is_connect):
    try:
        # This runs on the GPU worker
        print(f"Started processing: {pdf_path}")
        process_pdf(
            pdf_path=pdf_path,
            news_paper=news_paper,
            pdf_link=pdf_link,
            lang=lang,
            is_article=is_article,
            article_district=article_district,
            is_connect=is_connect
        )
        return "Success"
    except Exception as e:
        print(f"Error in task: {e}")
        return f"Failed: {e}"