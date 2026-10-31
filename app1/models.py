from django.db import models

# Create your models here.
from django.db import models

class GenomeUpload(models.Model):
    genome_file = models.FileField(upload_to="genomes/")
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Genome File {self.id}"
