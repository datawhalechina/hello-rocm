# 6.4 Preference Alignment with Reinforcement Learning

Before we get into the details of reinforcement learning, let us look at where it comes from. Reinforcement Learning (RL) is not actually a new idea. Its theoretical foundations can be traced back to behavioral psychology in the early 20th century, especially the studies on animal learning by Edward Thorndike and B. F. Skinner. Thorndike proposed the "Law of Effect": if a behavior leads to a positive result, the probability of repeating it increases. Skinner extended this idea further with operant conditioning, shaping behavior through rewards and punishments.

Reinforcement learning in computer science grew out of these psychological principles. In the 1980s, with the growth of compute power and mathematical theory, people began to apply these biological/psychological learning concepts to machines and computer programs, which led to modern reinforcement learning.

## 6.4.1 Basic Principles of Reinforcement Learning

Now let us get into the core part — the basic principles of RL.

- **State**: the concrete situation of a system at some moment. For example, in a board game, a state can be the current arrangement of all pieces on the board. For a self-driving car, a state may include the car's speed, position, and the positions of obstacles around it.
- **Action**: an action is something the agent can do in a given state. Taking a bicycle as an example, actions may include moving forward, stopping, turning, etc. In a complex system, the action set can be very large.
- **Reward**: feedback the agent receives after executing an action, usually a numeric value. Rewards can be immediate or delayed. A good action may receive a positive reward, a bad action a negative one.
- **Policy**: a set of rules that tells the agent how to choose actions. Simply put, the policy says what the agent should do in each state.
- **Value Function**: a tool to evaluate a policy, predicting the total reward obtainable in the long run from the current state. The value function helps the agent weigh short-term and long-term gains, not just the immediate step.
- **Model**: in some RL systems we build a model of the environment to help the agent foresee the consequences of its actions. This is useful in many complex computational settings.

![Reinforcement Learning](./images/7.1-1.png)

These elements work together so the agent can learn the best action strategy by trial and error in a virtual environment. In RL, the agent is the subject that learns and decides. It interacts with the environment in the following steps:

1. Observe the state: the agent first observes the current State.
2. Choose an action: based on the observed state and the predefined policy, the agent picks an Action.
3. Execute the action: the agent executes the chosen action.
4. Receive reward and new state: after execution, the agent receives the corresponding Reward and the updated new State from the environment.
5. Update the policy: the agent uses the obtained reward information to adjust its policy in order to obtain better results in the future.

This process is repeated, and the agent keeps optimizing its policy through repeated interaction, with the goal of performing better and better on the given task.

## 6.4.2 The Goal of Reinforcement Learning

The goal of RL is very explicit: ***through repeated trial and learning in a given environment, enable the agent to choose a sequence of actions that maximizes its total cumulative reward.*** This may sound abstract; we can use a game as analogy. In a game, the player aims to win high scores or finish levels through a series of operations (walking, jumping, fighting monsters, etc.). In RL, this notion of "high score / clearing levels" corresponds to "maximizing the reward".

Mathematically, the goal can be expressed as training a policy $\pi$ such that, in all states $s$, the actions chosen by the agent maximize the expected return $R(\tau)$. Specifically, we wish to maximize:

$$
E(R(\tau))_{\tau \sim P_{\theta}(\tau)} = \sum_{\tau} R(\tau) P_{\theta}(\tau)
$$

Where:
- $E(R(\tau))_{\tau \sim P_{\theta}(\tau)}$: the expected return $R(\tau)$ of trajectory $\tau$ under policy $P_{\theta}(\tau)$.
- $R(\tau)$: the return of trajectory $\tau$, i.e., the sum of all rewards from the initial state to the terminal state.
- $\tau$: a trajectory, i.e., a sequence of states and actions of the agent in the environment.
- $P_{\theta}(\tau)$: the probability of generating trajectory $\tau$ under parameters $\theta$, usually determined by the policy or policy network.
- $\theta$: the parameters of the policy, controlling the behavior of $P_{\theta}$.

To find this policy, we use gradient ascent and continuously update the policy parameters $\theta$ so that $E(R(\tau))_{\tau \sim P_{\theta}(\tau)}$ keeps increasing.

This learning paradigm is very effective because it does not depend on a large amount of labeled data; it learns by directly interacting with the environment and getting feedback. This makes RL show enormous potential in many adaptive, decision-making tasks, such as robotics control, autonomous driving, financial trading, and even games.

The application of RL in large models, such as AlphaGo and AlphaZero, further showcased the power of RL on complex tasks. These models continuously optimize their policies via RL, and eventually beat top human players in Go, chess, and similar games — demonstrating great potential of RL in complex tasks.

RL can also be used for preference alignment, e.g., letting a large model imitate human conversational style; it is also used in autonomous driving and many other fields. The application area of RL is very broad and there will be more application scenarios in the future.

## 6.4.3 Reward Model

In NLP, large language models (such as the LLaMA series, the Qwen series) have already shown strong text understanding and generation capabilities. However, these pre-trained models do not always directly meet specific business needs and human values. For this reason, people typically perform "Instruction Tuning" on pre-trained models — providing the model with specific prompts and examples so that it behaves more in line with human expectations on dialogue, QA, text generation, etc.

After initial instruction tuning, we still want the model's answers to be not only correct, but also to maximally satisfy human aesthetics, values, and safety standards. Hence the concept of Reinforcement Learning from Human Feedback (RLHF) was introduced. In RLHF, we first obtain human preference signals over model outputs (e.g., we present multiple model answers and let human annotators rank them), and then use these signals to guide model learning, gradually improving the alignment between model outputs and human preferences.

In order to automatically "score" (assign rewards to) model answers in the RLHF pipeline, we need to build a dedicated Reward Model. The reward model is trained on human-annotated data and, in actual deployment, scores model outputs automatically and independently, reducing the cost and latency of continuous human involvement.

## 6.4.4 Dataset Construction

Before building a Reward Model, we first need to prepare a high-quality human-feedback dataset. The core goal of this dataset is to provide multiple candidate completions for each given prompt, and to have human annotators carefully evaluate and rank these candidates. Through comparison and selection of answers, we provide the model with explicit reference standards, helping it learn how to generate outputs that better match human expectations on the given task.

Data can be collected by the following steps:

1. **Collect initial answers**: first, we generate multiple answers for a set of carefully designed prompts using a base "large model" that has gone through some basic fine-tuning (typically a pre-trained model with some instruction-following capability). These answers will serve as the base material for subsequent human annotation.

2. **Human annotation and evaluation**: with multiple candidate answers in hand, we invite professional annotators or crowdsourced workers to assess the quality of each answer. The evaluation is typically based on a set of pre-designed criteria — accuracy, completeness, contextual relevance, language fluency, and whether ethical/safety guidelines are followed. Comparing and ranking different answers helps us identify the best and worst, producing valuable training data.

3. **Data formatting and curation**: after annotation, we organize and format the data, typically using JSON, CSV, or another structured format that is convenient for computers. The dataset must clearly identify each prompt, its corresponding multiple completions, and the human annotators' choices (e.g., the best answer marked as "chosen" and the worse one marked as "rejected"). This labeling can be directly used as supervision signals for training the reward model, automatically biasing it toward generating high-quality answers.

Below is a simple data example showing two questions with their answers and human evaluation results. By comparing the "chosen" and "rejected" fields, we can intuitively see which answer is better.

```json
[
    {
        "question": "What is a list in Python?",
        "chosen": "A list in Python is an ordered, mutable container that allows storing multiple elements and supports index-based access.",
        "rejected": "A list in Python is used to store data."
    },
    {
        "question": "What is a tuple in Python?",
        "chosen": "A tuple in Python is an ordered, immutable container that allows storing multiple elements and cannot be modified once created.",
        "rejected": "A tuple in Python is used to store data."
    }
]
```

In the above example, the human annotators consider the "chosen" answer to be better than the "rejected" one in terms of description, accuracy, and information content. For instance, for the definition of a list, the "chosen" answer more clearly explains the characteristics of a list (ordered, mutable, supports indexed access), instead of just the vague "used to store data".


## 7.2.2 Reward Model Training

We can use the large-model RL framework TRL (Transformer Reinforcement Learning) to train the reward model. TRL is an RL-based training framework that aims to use human feedback to guide the model toward more human-aligned answers. In TRL, the reward model is treated as an independent component used to evaluate the model's outputs, and rewards or penalties are applied based on the evaluation result.
