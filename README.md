![figure1](https://github.com/user-attachments/assets/2023a5c1-a1eb-4e7b-8aff-b8fe0eacca76)Physically Consistent Sequence Generative Adversarial Networks for Wind Power Scenario Generation
==
Abstract
=
Scenario generation is an effective method for modeling the uncertainty of wind power output.
Traditional methods based on generative adversarial networks, while ensuring diversity of results, often produce scenarios that exceed physical limits due to inherent randomness.
This paper introduces a sequence generation adversarial network based on a physical model, effectively integrating data-driven outcomes with physical outcomes to enhance the realism and diversity of scenarios.
Building on this, to fully capture key dynamic features of wind power sequences, the model integrates Long Short-Term Memory networks with self-attention mechanisms at the data-driven layer,
enhancing the model’s ability to dynamically learn from wind power generation sequences.
To ensure stability in model training and accuracy in initial predictions, a historical trend learning unit has also been incorporated.
Finally, this paper evaluates the temporal correlation, accuracy, and stability of the generated scenarios on a real dataset and compares them with three existing methods to validate the effectiveness of the proposed approach.
Experimental results indicate that the proposed model achieves at least a 20% improvement in mean squared error and capability scores, demonstrating a distinct advantage in accurately depicting the realism of wind power scenarios compared to other methods.

