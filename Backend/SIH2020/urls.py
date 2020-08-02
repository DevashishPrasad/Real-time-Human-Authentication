from django.contrib import admin
from django.urls import include, path
from . import views
from django.conf import settings
from django.conf.urls.static import static
# from background.views import hello

urlpatterns = [
	path('',views.index,name="index"),
    path('background',include('background.urls'),name='background'),
    path('admin/', admin.site.urls),
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
# hello(repeat=1,repeat_until=None)