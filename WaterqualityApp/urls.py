from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from . import views

urlpatterns = [
		path("index.html", views.index, name="index"),
		path("Signup.html", views.Signup, name="Signup"),
		path("SignupAction", views.SignupAction, name="SignupAction"),	    	
		path("UserLogin.html", views.UserLogin, name="UserLogin"),
		path("UserLoginAction", views.UserLoginAction, name="UserLoginAction"),
		path("ProcessData", views.ProcessData, name="ProcessData"),
		path("TrainRF", views.TrainRF, name="TrainRF"),
		path("TrainLSTM", views.TrainLSTM, name="TrainLSTM"),
		path("Predict", views.Predict, name="Predict"),
		path("PredictAction", views.PredictAction, name="PredictAction"),
		path("UserScreen.html", views.index, name="UserScreen"),
		path('Feedback.html', views.Feedback, name='Feedback'),
        path('feedback/', views.Feedback, name='Feedback'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)