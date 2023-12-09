# GyML: Smart Fitness Trainer Using 3D Human Feedback Models
## Stanford CS 229 Final Project
Authors: Ishan Khare, Anthony Qin, Aditya Tadimeti <br>
A copy of our website's code can be found [in this repo](https://github.com/antqin/gyML). <br>
Here is a link to our final project paper: [click me!](https://stanford.edu/~iskhare/projects/GyML-CS229-paper.pdf)

# Introduction
Achieving and maintaining proper form during exercise is crucial for maximizing the benefits of physical workouts and minimizing the risk of injury. This challenge is prevalent among individuals of all fitness levels, from beginners to experts. Poor form and technique during lifting or exercise can increase the risk of injury and limit the overall benefits of the workout. To address this issue, we propose leveraging computer vision models that use state-of-the-art 3D human sensing to classify what exercise is being performed to provide immediate feedback on a user's exercise form. The feedback is presented in natural language, enabling users to make real-time adjustments. Given input in the form of a video of an exercise being performed, our algorithm performs Human Mesh Recovery (HMR) to reconstruct the human poses, classifies the exercise using PCA and linear regression, and then compares the HMR to a model video to output the predicted exercise along with feedback on the input video's form.

# Contributions
We worked on all aspects together and contributed evenly. <br>
**Ishan**: Dataset download and setup, multi-class logistic regression model, PCA and hyperparameter tuning, paper writing <br>
**Anthony**: Virtual machine setup, dataset splits, HMR, one-vs-rest logistic regression model, paper writing <br>
**Aditya**: Virtual machine setup, rep counting, statistical coach, frame analysis, paper writing
