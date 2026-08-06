from django import forms
class DocumentUploadForm(forms.Form):
    pdf = forms.FileField(widget=forms.ClearableFileInput(attrs={"accept": "application/pdf"}))
