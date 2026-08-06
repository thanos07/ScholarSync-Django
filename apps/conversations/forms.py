from django import forms
class QuestionForm(forms.Form):
    question = forms.CharField(max_length=2000, widget=forms.Textarea(attrs={"rows": 2, "placeholder": "Ask a question about your selected papers..."}))
