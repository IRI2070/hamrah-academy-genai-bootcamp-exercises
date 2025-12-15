<div dir="rtl">
  
  # گزارش فنی سامانه تحلیل احساسات کامنت‌های اسنپ‌فود  

## مقدمه  
هدف این کار، طراحی و پیاده‌سازی یک راه حل **تحلیل احساسات (Sentiment Analysis)** بر روی کامنت‌های کاربران اسنپ‌فود است. این سامانه با استفاده از روش‌های یادگیری ماشین و یادگیری عمیق، قادر است احساسات موجود در متن کامنت‌ها را در دسته‌های مختلف (مانند happy و sad) تشخیص دهد.  

## مدل‌های مورد استفاده  
سه مدل اصلی برای تحلیل احساسات توسعه داده شدند:  

1. **ترکیب TF-IDF و Logistic Regression**  
2. **مدل LSTM**  
3. **مدل Fine-Tuned بر پایه‌ی Tooka-BERT-Large**
## بررسی کیفیت دیتاست اولیه  
- دیتاست اصلی در [HuggingFace](https://huggingface.co/datasets/ParsiAI/snappfood-sentiment-analysis) منتشر شده است.  
- بررسی‌ها نشان داد که برخی برچسب‌ها (به‌ویژه happy و sad) به‌درستی تخصیص نیافته‌اند و متن کامنت با برچسب ناسازگار است.  

## اصلاح دیتاست  
برای رفع مشکل برچسب‌گذاری:  
- از مدل **GPT-4o-mini** استفاده شد تا دیتاست دوباره برچسب‌گذاری شود.  
- تنها رکوردهایی حفظ شدند که برچسب‌های جدید با برچسب‌های اصلی یکسان بودند.  
- متن تمامی کامنت‌ها با کتابخانه‌ی [DadmaTools](https://github.com/Dadmatech/DadmaTools) پاکسازی شد.  

### نتایج اصلاح دیتاست  
- **دقت مدل‌ها با دیتاست اصلی:** حدود 87٪  
- **دقت مدل‌ها با دیتاست اصلاح‌شده:** حدود 98٪  

## دسترسی به منابع پروژه  
- **دیتاست اصلاح‌شده:** [HuggingFace Dataset](https://huggingface.co/datasets/IRI2070/snappfood-refined-sentiment-dataset)  
- **مدل Fine-Tuned Tooka-BERT-Large:** [HuggingFace Model](https://huggingface.co/IRI2070/tooka-bert-large-sentiment)  
- **کدهای کامل پروژه:** [GitHub Repository](https://github.com/IRI2070/hamrah-academy-genai-bootcamp-exercises/tree/main/course-5-nlp-with-deep-learning/snappfood-sentiment-analysis)  

## جمع‌بندی  
این کار نشان داد که کیفیت دیتاست نقش کلیدی در عملکرد مدل‌های تحلیل احساسات دارد. با اصلاح برچسب‌ها و پاکسازی داده‌ها:  
- دقت مدل‌ها از 87٪ به 98٪ افزایش یافت.  
- مدل Fine-Tuned Tooka-BERT-Large بهترین عملکرد را ارائه داد.

</div>
