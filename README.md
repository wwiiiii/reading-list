# reading-list

This list of papers I read (or be going to read).
Most of papers are from ML area.

이때까지 읽은(읽을) 논문을 정리해둔 리스트입니다.
주로 머신러닝 분야에 집중되어 있습니다.

## Computer Vision
### Classification

#### Squeeze-and-Excitation Networks
- https://arxiv.org/abs/1709.01507
- 설명글: http://wwiiiii.tistory.com/entry/SqueezeandExcitation-Networks

### Interpretability

#### Visualizing and Understanding Convolutional Networks (ZFNet)
- https://arxiv.org/abs/1311.2901
- 설명 PPT: https://www.slideshare.net/ssuserb208cc1/cnn-visualization

#### Striving for Simplicity: The All Convolutional Net (Guided BackPropagation)
- https://arxiv.org/abs/1412.6806
- 설명 PPT: https://www.slideshare.net/ssuserb208cc1/cnn-visualization
- 구현 설명 PPT: https://www.slideshare.net/ssuserb208cc1/cnn-visualization-implementaion

#### Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization
- https://arxiv.org/abs/1610.02391
- 설명 PPT: https://www.slideshare.net/ssuserb208cc1/cnn-visualization
- 구현 설명 PPT: https://www.slideshare.net/ssuserb208cc1/cnn-visualization-implementaion

## Pairwise, Contrastive, Triplet Loss

기본 개념 소개 및 관련 논문 정리글: http://wwiiiii.tistory.com/entry/Pairwise-Triplet-Loss

### Pairwise
#### Improving Pairwise Ranking for Multi-label Image Classification
- https://arxiv.org/abs/1704.03135
- 설명글: http://wwiiiii.tistory.com/entry/Pairwise-Triplet-Loss
#### Who's Better? Who's Best? Pairwise Deep Ranking for Skill Determination
- https://arxiv.org/abs/1703.09913
- 설명글: http://wwiiiii.tistory.com/entry/Pairwise-Triplet-Loss

### Triplet

#### FaceNet: A Unified Embedding for Face Recognition and Clustering
- https://arxiv.org/abs/1503.03832
- 설명글: http://wwiiiii.tistory.com/entry/Pairwise-Triplet-Loss

#### In Defense of the Triplet Loss for Person Re-Identification
- https://arxiv.org/abs/1703.07737
- 설명글: http://wwiiiii.tistory.com/entry/Pairwise-Triplet-Loss

#### Beyond triplet loss: a deep quadruplet network for person re-identification
- https://arxiv.org/abs/1704.01719
- 설명글: http://wwiiiii.tistory.com/entry/Pairwise-Triplet-Loss

## Reinforcement Learning

#### Playing Atari with Deep Reinforcement Learning
- https://arxiv.org/abs/1312.56021
- 설명글: http://wwiiiii.tistory.com/entry/Deep-Q-Network
- 구현: https://github.com/wwiiiii/DQN/tree/master/breakout-dqn-2013

#### Human-level control through deep reinforcement learning
- https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf
- 설명글: http://wwiiiii.tistory.com/entry/Deep-Q-Network
- 구현: https://github.com/wwiiiii/DQN/tree/master/breakout_dqn_2015_nature

#### Policy Gradient Methods for Reinforcement Learning with Function Approximation (Policy Gradient)
- https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf
- 구현 (pytorch): https://gist.github.com/wwiiiii/0edb3feb6863c007e5927a8e4178a859
- 구현 (tensorflow): https://gist.github.com/wwiiiii/3c673d12e4092d50b90bf02d29f13ba4
- 설명글: https://www.slideshare.net/ssuserb208cc1/policy-gradient-120028773

## Multimodal Learning
#### FiLM: Visual Reasoning with a General Conditioning Layer
- https://arxiv.org/abs/1709.07871
- 요약: SENet과 유사하게, CNN에서 feature를 뽑는 과정에 text feature를 넣어 multimodal feature fusion.

#### Multimodal Compact Bilinear Pooling for Visual Question Answering and Visual Grounding
- https://arxiv.org/abs/1606.01847
- 요약: Compact Bilinear Pooling을 사용해서 multimodal feature fusion.

## Other ML related topics
#### Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
- http://proceedings.mlr.press/v37/ioffe15.html
- 설명글: http://wwiiiii.tistory.com/entry/Batch-Normalization
#### Ranking Measures and Loss Functions in Learning to Rank
- https://papers.nips.cc/paper/3708-ranking-measures-and-loss-functions-in-learning-to-rank.pdf
- 요약: 자주 쓰이는 Ranking loss 들이 실제로 Ranking Metric인 MAP/NDCG을 개선하는지에 대한 논문. 결론은 loss에 대해 optimize 해도 metric이 개선됨을 증명함.
#### Multi-armed bandit
- 설명글: http://wwiiiii.tistory.com/entry/MultiArmed-Bandit-Problem
- 구현: https://github.com/wwiiiii/Multi-Armed-Bandit
