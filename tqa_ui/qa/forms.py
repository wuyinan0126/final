from django import forms
from django.conf import settings
from django.forms import TextInput

from qa.models import Question
from tqa_ui.tqa.core.tqa_core import TqaThread


class QuestionForm(forms.ModelForm):
    """ 提问表单 """

    class Meta:
        model = Question
        fields = [
            'title',
            'description',
            'tags',
        ]

    def __init__(self, *args, **kwargs):
        super(QuestionForm, self).__init__(*args, **kwargs)
        try:
            settings.QA_SETTINGS['qa_description_optional']
            self.fields['description'].required = not settings.QA_SETTINGS['qa_description_optional']
            # 标题栏宽度
            self.fields['title'].widget = TextInput(attrs={'size': 72, 'maxlength': 70})
            # 标签栏宽度
            self.fields['tags'].widget = TextInput(attrs={'size': 66, 'maxlength': 88})
            # 标签栏是否强制
            self.fields['tags'].required = False
            print(dir(self.fields['tags']))
        except KeyError:
            pass

    def clean_tags(self):
        """ 对form中的tags列表中的每个元素进行操作 """
        title = self.cleaned_data['title']
        description = self.cleaned_data['description']
        tags = self.cleaned_data['tags']
        if tags:
            return tags
        return TqaThread.get_tags(title, description)
