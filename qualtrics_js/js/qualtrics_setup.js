var conditionKey = 'condition_#_#'; // Declare conditionKey outside the functions

Qualtrics.SurveyEngine.addOnload(function() {
    console.log("Qualtrics onload started with conditionKey:", conditionKey);
    executeCustomSurveyLogic(this, conditionKey);
});

Qualtrics.SurveyEngine.addOnReady(function()
{
    console.log("Page is fully displayed and ready.");
});

Qualtrics.SurveyEngine.addOnUnload(function()
{
    console.log("Page is about to be unloaded.");
});

// Add OnPageSubmit to trigger recodeAndStoreChoice function after respondent submits their choice
Qualtrics.SurveyEngine.addOnPageSubmit(function() {
    console.log("Page submit triggered for conditionKey:", conditionKey);
    recodeAndStoreChoice(this, conditionKey); // Call your function here
});
