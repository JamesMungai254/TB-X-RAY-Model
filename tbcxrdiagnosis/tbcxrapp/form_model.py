from django.db import models

class PatientsModel(models.Model):
    id = models.AutoField(primary_key=True)
    p_name = models.CharField(max_length=100)
    p_status = models.CharField(max_length=100)
    p_address = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return self.p_name, self.p_status, self.p_address
