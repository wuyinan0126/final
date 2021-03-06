from annoying.fields import AutoOneToOneField
from django.conf import settings
from django.db import models
from django.db.models import F
from django.forms import TextInput
from django.utils.encoding import python_2_unicode_compatible
from django.utils.text import slugify
from django_markdown.models import MarkdownField
from hitcount.models import HitCountMixin
from taggit.managers import TaggableManager
from taggit.models import Tag, TaggedItem


@python_2_unicode_compatible
class UserQAProfile(models.Model):
    """Model class to define a User profile for the app, directly linked
    to the core Django user model."""
    user = AutoOneToOneField(settings.AUTH_USER_MODEL, primary_key=True,
                             on_delete=models.CASCADE)
    points = models.IntegerField(default=0)
    # The additional attributes we wish to include.
    website = models.URLField(blank=True)

    def modify_reputation(self, added_points):
        """Core function to modify the reputation of the user profile."""
        self.points = F('points') + added_points
        self.save()

    def __str__(self):  # pragma: no cover
        return self.user.username


# ------------------------------------------------------------------------------
# 让django-taggit支持slug unicode
# ------------------------------------------------------------------------------
class MyTag(Tag):
    class Meta:
        proxy = True

    def slugify(self, tag, i=None):
        return slugify(tag, allow_unicode=True)


class MyTaggedItem(TaggedItem):
    class Meta:
        proxy = True

    @classmethod
    def tag_model(cls):
        return MyTag


# ------------------------------------------------------------------------------
# 自动回答线程 End
# ------------------------------------------------------------------------------

@python_2_unicode_compatible
class Question(models.Model, HitCountMixin):
    """Model class to contain every question in the forum"""
    slug = models.SlugField(max_length=200, allow_unicode=True)
    title = models.CharField(
        max_length=200, blank=False, verbose_name="问题标题 (请简要概括问题)",
    )
    description = MarkdownField(verbose_name="问题描述 (请详细描述问题，支持markdown语法)")
    pub_date = models.DateTimeField('date published', auto_now_add=True)
    tags = TaggableManager(
        through=MyTaggedItem,
        verbose_name="标签 (以逗号分割，推荐自动生成!)", help_text="",
    )
    reward = models.IntegerField(default=0)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    closed = models.BooleanField(default=False)
    positive_votes = models.IntegerField(default=0)
    negative_votes = models.IntegerField(default=0)
    total_points = models.IntegerField(default=0)

    def save(self, *args, **kwargs):
        if not self.id:
            self.slug = slugify(self.title, allow_unicode=True)
            try:
                points = settings.QA_SETTINGS['reputation']['CREATE_QUESTION']

            except KeyError:
                points = 0

            self.user.userqaprofile.modify_reputation(points)

        self.total_points = self.positive_votes - self.negative_votes
        super(Question, self).save(*args, **kwargs)

    def __str__(self):
        return str(self.id)


@python_2_unicode_compatible
class Answer(models.Model):
    """Model class to contain every answer in the forum and to link it
    to the proper question."""
    question = models.ForeignKey(Question, on_delete=models.CASCADE)
    answer_text = MarkdownField()
    pub_date = models.DateTimeField('date published', auto_now_add=True)
    updated = models.DateTimeField('date updated', auto_now=True)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    answer = models.BooleanField(default=False)
    positive_votes = models.IntegerField(default=0)
    negative_votes = models.IntegerField(default=0)
    total_points = models.IntegerField(default=0)

    def save(self, *args, **kwargs):
        try:
            points = settings.QA_SETTINGS['reputation']['CREATE_ANSWER']

        except KeyError:
            points = 0

        self.user.userqaprofile.modify_reputation(points)
        self.total_points = self.positive_votes - self.negative_votes
        super(Answer, self).save(*args, **kwargs)

    def __str__(self):  # pragma: no cover
        return self.answer_text

    class Meta:
        ordering = ['-answer', '-pub_date']


class VoteParent(models.Model):
    """Abstract model to define the basic elements to every single vote."""
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    value = models.BooleanField(default=True)

    class Meta:
        abstract = True


class AnswerVote(VoteParent):
    """Model class to contain the votes for the answers."""
    answer = models.ForeignKey(Answer, on_delete=models.CASCADE)

    class Meta:
        unique_together = (('user', 'answer'),)


class QuestionVote(VoteParent):
    """Model class to contain the votes for the questions."""
    question = models.ForeignKey(Question, on_delete=models.CASCADE)

    class Meta:
        unique_together = (('user', 'question'),)


@python_2_unicode_compatible
class BaseComment(models.Model):
    """Abstract model to define the basic elements to every single comment."""
    pub_date = models.DateTimeField('date published', auto_now_add=True)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)

    class Meta:
        abstract = True

    def __str__(self):  # pragma: no cover
        return self.comment_text


class AnswerComment(BaseComment):
    """Model class to contain the comments for the answers."""
    comment_text = MarkdownField()
    answer = models.ForeignKey(Answer, on_delete=models.CASCADE)

    def save(self, *args, **kwargs):
        try:
            points = settings.QA_SETTINGS['reputation']['CREATE_ANSWER_COMMENT']

        except KeyError:
            points = 0

        self.user.userqaprofile.modify_reputation(points)
        super(AnswerComment, self).save(*args, **kwargs)


class QuestionComment(BaseComment):
    """Model class to contain the comments for the questions."""
    comment_text = models.CharField(max_length=250)
    question = models.ForeignKey(Question, on_delete=models.CASCADE)

    def save(self, *args, **kwargs):
        try:
            points = settings.QA_SETTINGS['reputation']['CREATE_QUESTION_COMMENT']

        except KeyError:
            points = 0

        self.user.userqaprofile.modify_reputation(points)
        super(QuestionComment, self).save(*args, **kwargs)
