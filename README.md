# MusicBackend

This application is a RESTful API functioning as the backend for a music app that generates song recommendations based on a user's preferences. The front end for
this app (developed in Flutter) can be found [here](https://github.com/dad9489/Music-Flutter).

This API is deployed using AWS Elastic
Beanstalk and is running at http://musicbackend-dev.us-east-1.elasticbeanstalk.com/

(This API is currently under development and is therefore not complete in indended functionality)

## Some notes:
- The configuration variables are stored in the .env file. Change the values in this file to configure the application to
run locally. If deploying to AWS, the config variables can be changed for the Elastic Beanstalk instance in
Configuration -> Software.
    - To remove the file from git so the local changes are not tracked, run `git update-index --assume-unchanged .env`.
    To undo this, run `git update-index --no-assume-unchanged .env`
- To connect to Google Sheets API, place `credentials.json` file in the main directory.