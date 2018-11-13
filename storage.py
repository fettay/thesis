from globals import BUCKET_NAME
import boto3
from io import BytesIO


class Storage:
    def __init__(self, bucket=BUCKET_NAME):
        self._bucket_name = bucket
        self._bucket = self.get_bucket()

    def get_bucket(self):
        client = boto3.resource('s3')
        return client.Bucket(self._bucket_name)

    def put(self, key, value):
        return self._bucket.put_object(Key=key, Body=value)

    def get(self, key):
        io = BytesIO()
        self._bucket.download_fileobj(key, io)
        io.seek(0)
        return io.read()

    def delete(self, key):
        return self._bucket.delete_objects(
            Delete={
                'Objects': [
                    {
                        'Key': key
                    }
                ]})

    def ls(self):
        for obj in self._bucket.objects.all():
            yield obj.key


def test():
    storage = Storage()
    storage.put('test', b'a')
    print(storage.get('test'))
    storage.delete('test')

