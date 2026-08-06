from django.contrib.auth.models import User
from django.test import TestCase
from django.urls import reverse
from apps.workspaces.models import Workspace
class WorkspacePermissionTests(TestCase):
    def test_other_users_workspace_is_not_visible(self):
        owner=User.objects.create_user("owner",password="test-pass-123")
        intruder=User.objects.create_user("intruder",password="test-pass-123")
        workspace=Workspace.objects.create(owner=owner,name="Private")
        self.client.login(username="intruder",password="test-pass-123")
        response=self.client.get(reverse("workspace-detail",args=[workspace.id]))
        self.assertEqual(response.status_code,404)
