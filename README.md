# RL-Reinforce

强化学习-Reinforce 调研与实现

## Log

---

### 2021/6/15

* [ ] 调研-优势函数
  * 优势函数其实就是将Q-Value“归一化”到Value baseline上，如上讨论的，这样有助于提高学习效率，同时使学习更加稳定；同时经验表明，优势函数也有助于减小方差，而方差过大导致过拟合的重要因素。
* [ ] 调研-important sampling