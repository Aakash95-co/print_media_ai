import pyodbc

SQL_SERVER_CONN_STR = (
    "DRIVER={ODBC Driver 18 for SQL Server};"
    "SERVER=172.17.28.179;"
    "DATABASE=AI_NewsAnalysis;"
    "UID=linq-usr;"
    "PWD=linq@usr145;"
    "Encrypt=no;" 
)

def insert_news_analysis_entry(article_info_insert):
    """Call stored procedure [dbo].[NewsAnalysisEntry_Insert]."""
    try:
        conn = pyodbc.connect(SQL_SERVER_CONN_STR, timeout=10)
        cursor = conn.cursor()

        cursor.execute("""
            EXEC [dbo].[NewsAnalysisEntry_Insert]
                @Page_id = ?,            -- 1
                @Article_id = ?,         -- 2
                @Newspaper_name = ?,     -- 3
                @Article_link = ?,       -- 4
                @Gujarati_Text = ?,      -- 5
                @English_Text = ?,       -- 6
                @Text_Sentiment = ?,     -- 7
                @Is_govt = ?,            -- 8
                @Category = ?,           -- 9
                @Prabhag = ?,            -- 10
                @District = ?,           -- 11
                @Dcode = ?,              -- 12
                @Tcode = ?,              -- 13
                @Cat_code = ?,           -- 14
                @Title = ?,              -- 15
                @PrabhagId  = ?,          -- 16
                @AI_ID = ?                -- 17       
        """, article_info_insert)

        conn.commit()
        cursor.close()
        conn.close()
        print(f"✅ Stored procedure executed successfully for Article ID: {article_info_insert[1]}")

    except Exception as e:
        print(f"❌ Failed to insert via stored procedure: {e}")
