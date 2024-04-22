# forms.py
from django import forms
class FeedbackForm(forms.Form):
    feedback_text = forms.CharField(widget=forms.Textarea(attrs={'rows': 5, 'cols': 40}), label='Enter your feedback:')
