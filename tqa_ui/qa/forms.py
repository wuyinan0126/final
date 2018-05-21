import unidecode as unidecode
from django import forms
from django.conf import settings
from qa.models import Question


class QuestionForm(forms.ModelForm):
    class Meta:
        model = Question
        fields = ['title', 'description', 'tags']

    def __init__(self, *args, **kwargs):
        super(QuestionForm, self).__init__(*args, **kwargs)

        try:
            settings.QA_SETTINGS['qa_description_optional']
            self.fields['description'].required = not settings.QA_SETTINGS['qa_description_optional']

        except KeyError:
            pass

    def clean_tags(self):
        """ 对form中的tags列表中的每个元素进行操作 """
        data = self.cleaned_data['tags']
        return data
