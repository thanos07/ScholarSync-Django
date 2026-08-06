from django.contrib import admin
from .models import Conversation, Message, Citation
admin.site.register(Conversation)
admin.site.register(Message)
admin.site.register(Citation)
