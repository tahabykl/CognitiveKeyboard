# CognitiveKeyboard Android App

The android app is what allows us to record the patient keystrokes. To make sure user data stays private and secure, we perform the operations that use PII (personally identifiable information) inside the user device.  

For other operations that do not require PII, we perform the following privacy-preserving operations in user's local device:

1) Typing Speed 
2) Inter-keystroke Interval  - Time Delay between two successive key presses
3) Typing Accuracy -  The frequency of backspace 
4) Character Variability - Frequency of different characters being typed.
5) Special Character Usage -  How often non-alphabetic characters (e.g., punctuation, numbers) are used.

These operations are performed every 30 seconds on the words that were typed in the last 30 seconds. 

After the operations are performed, data is sent to the API through the following endpoint:

(TODO add the relevant API endpoint here)

We are planning to do sentiment analysis as well. Locally. On the whole text. Because it can catch easily identfiable (but possibly not involving any of the cognitive decline parameters listed above). These statements may look like the following:

"I will **** ****"
"How to **** **** ****"

These are obviously very easy to identify with human eyes, but not with the cognitive decline parameters listed above. Thus, we need to run sentiment analysis in local.

---

How to run the app:

1) Open it. It will direct you to settings, where you have to 

TODO continue
