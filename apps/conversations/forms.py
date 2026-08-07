from django import forms


class QuestionForm(forms.Form):
    question = forms.CharField(
        max_length=2000,
        widget=forms.Textarea(
            attrs={
                "rows": 2,
                "placeholder": "Ask about your uploaded papers, request a summary, table, comparison, or formula...",
                "data-question-input": "",
                "aria-label": "Research question",
            }
        ),
    )
