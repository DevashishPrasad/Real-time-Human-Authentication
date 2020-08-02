from django.contrib import admin
from django.urls import include, path
from . import views
from django.conf import settings
from django.conf.urls.static import static
from background.views import hello

urlpatterns = [
	path('',views.main),
    # path('background',include('background.urls'),name='background'),
    # path('admin/', admin.site.urls),
]
hello(repeat=1,repeat_until=None)   