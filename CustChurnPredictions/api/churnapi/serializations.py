from rest_framework import serializers
from api.churnapi.models import UserRegistrationModel

class UserRegisterSerializer(serializers.ModelSerializer):
    #def validate(self, data):
        #if data['name'] !='Dell':
            #raise serializers.ValidationError("Name Must Be Dell")
        #return data
    class Meta:
        fields = '__all__'
        model = UserRegistrationModel
