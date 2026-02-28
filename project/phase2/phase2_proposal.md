# CME 241 Project Proposal
Jordan Rothkowitz

Winter 2026

Project Title: Multi-Capital Decision-Making Under Energy Constraints

## Problem Statement
This project addresses a very general, idealized life optimization problem faced by early-career professionals who are also competitive athletes. Specifically, I focus on a young professional (myself; age 22-30) moving to the Main Line suburbs of Philadelphia to start a new job in Malvern while maintaining a serious weightlifting training program. The challenge is optimizing sequential decisions about housing, training, investments, and consumption to maximize lifetime utility under constraints of limited financial capital, time, finite energy capacity, and stochastic work demands.

Unlike traditional personal finance optimization problems that focus solely on wealth maximization, this formulation recognizes that:
1. Energy is a scarce resource: Physical and mental energy are depleted by work, commuting, and training, and must be actively recovered
2. Money does not necessarily correspond to life quality: Sometimes financially "suboptimal" choices (expensive apartment near gym) yield better life outcomes
3. Training provides direct utility: Athletic performance and training quality are primary sources of life satisfaction, not just costs to be minimized
 
This problem involves allocating limited resources across:
- Wealth Building: Investments, emergency fund, retirement savings
- Location Quality: Housing cost vs. commute time vs. neighborhood amenities (nightlife, cafes, walkability)
- Athletic Performance: Training intensity, coaching, nutrition, recovery vs. injury risk
- Life Enjoyment: Discretionary consumption, social activities, experiences
- Energy Management: Balancing training capacity with stochastic work demands and commute time

---

## Practical Relevance

This problem is relevant for several reasons:

- Immediate Personal Application: I am currently facing this exact decision problem as I prepare to move to the Main Line suburbs for work in Malvern
- Broad Demographic: Many young professionals face similar tradeoffs between career, fitness, location, and lifestyle
- Understudied in RL Literature: Most finance RL focuses on pure portfolio optimization; few incorporate energy constraints and multi-capital optimization
- Pedagogically Rich: Very complicated, idealized problem demonstrates curse of dimensionality, curse of modeling, and the value of RL over classical DP. The problem will need to be greatly simplified in each stage to be tractable.

---

## Version 1: Ideal Version

### MDP Specification: $\mathcal{M}_1 = (S_1, A_1, P_1, R_1, \gamma)$

### State Space $S_1$
Continuous, 47-dimensional state space capturing rich detail:

**Financial State** (7 vars):
- $W_t \in [0, \infty)$: Liquid wealth (checking/savings, $)
- $E_t \in [0, \infty)$: Emergency fund ($)
- $p_t^{stocks} \in [0, \infty)$: Stock holdings ($)
- $p_t^{bonds} \in [0, \infty)$: Bond holdings ($)
- $r_t^{401k} \in [0, \infty)$: 401k balance ($)
- $r_t^{IRA} \in [0, \infty)$: IRA balance ($)
- $D_t \in [0, \infty)$: Outstanding debt ($)
- $S_t \in [60000, 200000]$: Annual salary ($)
- $B_t \in [0, 50000]$: Expected annual bonus ($)

**Housing/Location State** (17 vars):
- $x_t \in [39.95, 40.10]$: Latitude coordinate in Main Line suburbs (Malvern area)
- $y_t \in [-75.60, -75.35]$: Longitude coordinate
- $h_t^{rent} \in [1000, 3500]$: Monthly rent ($)
- $h_t^{size} \in [400, 1500]$: Apartment size (sq ft)
- $h_t^{bedrooms} \in \{0, 1, 2\}$: Number of bedrooms (studio, 1br, 2br)
- $h_t^{quality} \in [0, 100]$: Overall quality index (fixtures, finishes, maintenance)
- $h_t^{noise} \in [0, 100]$: Noise level (0=quiet, 100=very loud)
- $h_t^{laundry} \in \{0, 1\}$: In-unit laundry (binary)
- $h_t^{storage} \in \{0, 1\}$: Bike/ski storage available (binary)
- $n_t^{safety} \in [0, 100]$: Neighborhood safety score
- $n_t^{walkability} \in [0, 100]$: Walk score
- $n_t^{nightlife} \in [0, 100]$: Nightlife density/quality
- $n_t^{cafes} \in [0, 100]$: Coffee shop density/quality
- $n_t^{parks} \in [0, 100]$: Access to green space
- $c_t^{work} \in [5, 90]$: Commute time to work (minutes)
- $c_t^{gym} \in [5, 60]$: Distance to gym (minutes)
- $c_t^{grocery} \in [2, 30]$: Distance to grocery store (minutes)

**Work State** (2 vars):
- $W_t^{intensity} \in [0, 100]$: Work intensity/demand (stochastic, exogenous)
- $j_t^{performance} \in [0, 100]$: Job performance score

**Energy/Capacity State** (7 vars):
- $\Phi_t \in [0, 100]$: Physical energy level
- $\Psi_t \in [0, 100]$: Mental energy level
- $\Theta_t \in [0, 100]$: Accumulated training fatigue
- $Q_t \in [0, 100]$: Sleep quality index
- $\Nu_t \in [0, 100]$: Nutrition quality index
- $\Gamma_t \in [0, 100]$: Injury severity (0=healthy, 100=severely injured)
- $\Sigma_t \in [0, 100]$: Stress level

**Training/Performance State** (7 vars):
- $m_t^{snatch} \in [40, 140]$: Snatch 1RM (kg)
- $m_t^{C\&J} \in [50, 180]$: Clean & Jerk 1RM (kg)
- $m_t^{squat} \in [80, 250]$: Back squat 1RM (kg)
- $v_t^{mileage} \in [0, 100]$: Recent running/walking mileage (miles/week, 8-week EWMA)
- $v_t^{load} \in [0, 500]$: Recent training load (tons/week, 8-week EWMA)
- $v_t^{intensity} \in [100, 180]$: Average HR per session (bpm, 8-week EWMA)
- $F_t \in [0, 26]$: Time until next competition (biweeks)

**Market/Economic State** (6 vars):
- $\mu_t^{stock} \in [-0.20, 0.30]$: Expected stock return (annualized)
- $\mu_t^{bond} \in [-0.05, 0.10]$: Expected bond return (annualized)
- $r_t^{rf} \in [0, 0.06]$: Risk-free rate (annualized)
- $\pi_t \in [-0.02, 0.08]$: Inflation rate (annualized)
- $X_t^{labor} \in [0, 100]$: Labor market strength for early-career professionals

**Time** (1 var):
- $\tau_t \in \{0, 1, ..., 77\}$: Biweek (0 = start, 77 = end of 3-year horizon)

**Total: 47 state dimensions**

### Action Space $A_1$
Continuous and discrete, 21-dimensional action space:

**Housing** (4 actions, yearly decisions):
- $a_t^{move} \in \{0, 1\}$: Whether to move apartments
- $a_t^{location} = (x_t^{new}, y_t^{new}) \in \mathbb{R}^2$: New location coordinates (if moving)
- $a_t^{rent\_budget} \in [1000, 3500]$: Monthly rent budget (if moving)
- $a_t^{lease} \in \{6, 12, 18\}$: Lease length in months (if moving)

**Financial** (7 actions, biweekly decisions):
- $a_t^{invest} \in [0, S_t/26]$: Investment contribution this biweek
- $\alpha_t^{stocks} \in [0, 1]$: Portfolio allocation to stocks (rest to bonds)
- $a_t^{401k} \in [0, \min(0.20 \cdot S_t/26, 23000/26)]$: 401k contribution (pre-tax, employer match)
- $a_t^{IRA} \in [0, 7000/26]$: IRA contribution (post-tax)
- $a_t^{emergency} \in [0, S_t/26]$: Emergency fund contribution
- $a_t^{debt} \in [0, W_t]$: Debt repayment amount
- $a_t^{rebalance} \in \{0, 1\}$: Rebalance portfolio to target allocation

**Training** (5 actions, biweekly decisions):
- $a_t^{training\_intensity} \in [0, 100]$: Training program intensity (0=rest, 100=max)
- $a_t^{coaching} \in \{0, 1, 2\}$: Coaching tier (0=self, 1=online, 2=in-person)
- $a_t^{nutrition} \in [50, 500]$: Nutrition budget per biweek ($)
- $a_t^{recovery} \in [0, 200]$: Recovery investment (massage, PT, etc., $ per biweek)
- $a_t^{compete} \in \{0, 1\}$: Compete in upcoming competition (binary)

**Consumption/Lifestyle** (5 actions, biweekly decisions):
- $a_t^{dining} \in [0, 500]$: Dining out budget per biweek ($)
- $a_t^{social} \in [0, 300]$: Social activities budget (bars, events, $ per biweek)
- $a_t^{hobbies} \in [0, 200]$: Hobbies/entertainment budget ($ per biweek)
- $a_t^{travel} \in \{0, 1\}$: Take a trip this quarter (binary, avg $1500/trip)

**Total: 21 action dimensions** (4 housing + 7 financial + 5 training + 5 consumption)

### Transition Dynamics $P_1(s_{t+1} | s_t, a_t)$

**Financial Transitions**:

$$W_{t+1} = W_t + \frac{S_t}{26} - \frac{h_t^{rent}}{26} - a_t^{invest} - a_t^{401k} - a_t^{IRA} - a_t^{emergency} - a_t^{debt}$$
$$- a_t^{dining} - a_t^{social} - a_t^{hobbies} - 1500 \cdot a_t^{travel} + \epsilon_t^W$$

where $\epsilon_t^W \sim \mathcal{N}(0, 100^2)$ captures unexpected expenses/income.

$$E_{t+1} = E_t + a_t^{emergency}$$

$$p_{t+1}^{stocks} = p_t^{stocks} + \alpha_t^{stocks} \cdot a_t^{invest} + r_{t+1}^{stocks} \cdot p_t^{stocks}$$

$$p_{t+1}^{bonds} = p_t^{bonds} + (1 - \alpha_t^{stocks}) \cdot a_t^{invest} + r_{t+1}^{bonds} \cdot p_t^{bonds}$$

where $r_{t+1}^{stocks} \sim \mathcal{N}(\mu_t^{stock}/26, (0.18/\sqrt{26})^2)$ and $r_{t+1}^{bonds} \sim \mathcal{N}(\mu_t^{bond}/26, (0.06/\sqrt{26})^2)$ are biweekly returns.

$$r_{t+1}^{401k} = r_t^{401k} + a_t^{401k} \cdot (1 + \text{match\_rate}) + r_{t+1}^{port} \cdot r_t^{401k}$$

$$r_{t+1}^{IRA} = r_t^{IRA} + a_t^{IRA} + r_{t+1}^{port} \cdot r_t^{IRA}$$

where $r_{t+1}^{port} = 0.7 \cdot r_{t+1}^{stocks} + 0.3 \cdot r_{t+1}^{bonds}$ (target allocation).

$$D_{t+1} = \max(0, D_t \cdot (1 + r_t^{debt}/26) - a_t^{debt})$$

**Salary Transitions**:

$$S_{t+1} = \begin{cases}
S_t \cdot (1 + 0.15 + \epsilon_t^{promo}) & \text{if promoted} \\
S_t \cdot (1 + 0.03 + \epsilon_t^{raise}) & \text{annual raise (every 26 biweeks)} \\
S_t & \text{otherwise}
\end{cases}$$

where $P(\text{promoted}) = \text{logistic}(-3 + 0.03 \cdot j_t^{performance} + 0.02 \cdot \tau_t)$ and $\epsilon_t^{promo}, \epsilon_t^{raise} \sim \mathcal{N}(0, 0.02^2)$.

**Housing Transitions**:

If $a_t^{move} = 1$:
- $(x_{t+1}, y_{t+1}) = a_t^{location}$
- $h_{t+1}^{rent} = a_t^{rent\_budget}$
- Other housing characteristics $(h_{t+1}^{size}, h_{t+1}^{quality}, ...)$ sampled from distribution conditioned on location and rent:
  $$P(H_{t+1} | x_{t+1}, y_{t+1}, h_{t+1}^{rent})$$
- Neighborhood characteristics are deterministic functions of location:
  $$n_{t+1}^{safety} = f_{safety}(x_{t+1}, y_{t+1}), \quad n_{t+1}^{walkability} = f_{walk}(x_{t+1}, y_{t+1}), ...$$
- Commute times are deterministic functions of location and work/gym locations:
  $$c_{t+1}^{work} = \text{distance}((x_{t+1}, y_{t+1}), (x_{work}, y_{work})), ...$$

If $a_t^{move} = 0$:
- All housing/location variables remain unchanged
- $h_{t+1}^{rent} = h_t^{rent} \cdot (1 + 0.03 + \epsilon_t^{rent})$ (annual rent increase)

**Work Intensity Transition** (exogenous, stochastic):

$$W_{t+1}^{intensity} = 0.7 \cdot W_t^{intensity} + 0.3 \cdot 50 + \epsilon_t^{work}$$

where $\epsilon_t^{work} \sim \mathcal{N}(0, 15^2)$ (mean-reverting AR(1) around 50).

**Job Performance Transition**:

$$j_{t+1}^{performance} = \text{clip}(0.6 \cdot j_t^{performance} + 0.25 \cdot \Psi_t - 0.15 \cdot \Sigma_t + 0.1 \cdot \min(\tau_t, 52) + \epsilon_t^{perf}, 0, 100)$$

where $\epsilon_t^{perf} \sim \mathcal{N}(0, 8^2)$. Performance depends on mental energy, stress, tenure, and random factors.

**Energy Transitions**:

$$\Phi_{t+1} = \text{clip}(\Phi_t - 0.4 \cdot W_t^{intensity}/100 - 0.35 \cdot v_t^{load}/500 - 0.02 \cdot (c_t^{work} + c_t^{gym})$$
$$+ 25 \cdot \mathbb{1}_{Q_t > 80} + 15 \cdot \mathbb{1}_{\Nu_t > 75} - 10 \cdot \Gamma_t/100 + \epsilon_t^\Phi, 0, 100)$$

where $\epsilon_t^\Phi \sim \mathcal{N}(0, 5^2)$.

$$\Psi_{t+1} = \text{clip}(\Psi_t - 0.5 \cdot W_t^{intensity}/100 - 0.3 \cdot \Sigma_t - 0.01 \cdot (c_t^{work} + c_t^{gym})$$
$$+ 20 \cdot \mathbb{1}_{Q_t > 80} + 0.15 \cdot n_t^{walkability} + \epsilon_t^\Psi, 0, 100)$$

where $\epsilon_t^\Psi \sim \mathcal{N}(0, 6^2)$.

$$\Theta_{t+1} = \text{clip}(\Theta_t + v_t^{load}/100 + v_t^{intensity}/10 - \text{recovery}(\Nu_t, Q_t, a_t^{recovery}) + \epsilon_t^\Theta, 0, 100)$$

where recovery function: $\text{recovery}(\Nu, Q, r) = 0.3 \cdot \Nu/100 + 0.4 \cdot Q/100 + 0.01 \cdot r$.

$$Q_{t+1} = \text{clip}(60 - 0.3 \cdot h_t^{noise} - 0.2 \cdot \Sigma_t - 0.15 \cdot \Theta_t + 0.1 \cdot n_t^{parks} + \epsilon_t^Q, 0, 100)$$

$$\Nu_{t+1} = \text{clip}(40 + 0.12 \cdot a_t^{nutrition} - 0.2 \cdot W_t^{intensity}/100 + \epsilon_t^\Nu, 0, 100)$$

$$\Gamma_{t+1} = \max(0, \Gamma_t - 5 + 20 \cdot \mathbb{1}_{\text{injury event}} + \epsilon_t^\Gamma)$$

where $P(\text{injury event}) = \text{logistic}(-6 + 0.05 \cdot \Theta_t + 0.02 \cdot v_t^{load} - 0.03 \cdot a_t^{recovery})$.

$$\Sigma_{t+1} = \text{clip}(0.6 \cdot \Sigma_t + 0.3 \cdot W_t^{intensity}/100 + 20 \cdot \mathbb{1}_{W_t < 5000} + 15 \cdot \mathbb{1}_{E_t < 3 \cdot h_t^{rent}}$$
$$- 0.1 \cdot n_t^{nightlife} - 0.05 \cdot n_t^{cafes} + \epsilon_t^\Sigma, 0, 100)$$

**Training Volume Transitions** (EWMA of recent 8 weeks):

$$v_{t+1}^{mileage} = 0.85 \cdot v_t^{mileage} + 0.15 \cdot \text{mileage}(a_t^{training\_intensity}, \Phi_t)$$

$$v_{t+1}^{load} = 0.85 \cdot v_t^{load} + 0.15 \cdot \text{load}(a_t^{training\_intensity}, \Phi_t, a_t^{coaching})$$

$$v_{t+1}^{intensity} = 0.85 \cdot v_t^{intensity} + 0.15 \cdot \text{HR}(a_t^{training\_intensity}, \Phi_t)$$

where current training depends on intensity choice and available energy.

**Strength Transitions**:

$$m_{t+1}^{snatch} = m_t^{snatch} + \text{adaptation}(v_t^{load}, a_t^{coaching}, \Theta_t, \Phi_t, \Nu_t, \Gamma_t) + \epsilon_t^{snatch}$$

where $\text{adaptation}(...) = (0.005 \cdot v_t^{load} + 0.2 \cdot (a_t^{coaching} - 0.5)) \cdot (1 - \Theta_t/100) \cdot (\Phi_t/100)^{0.5} \cdot (\Nu_t/100) \cdot (1 - \Gamma_t/100)$.

Similar expressions for $m_{t+1}^{C\&J}$ and $m_{t+1}^{squat}$.

**Market Transitions**:

$$\mu_{t+1}^{stock} = 0.8 \cdot \mu_t^{stock} + 0.2 \cdot 0.08 + \epsilon_t^{\mu_{stock}}$$

$$\mu_{t+1}^{bond} = 0.85 \cdot \mu_t^{bond} + 0.15 \cdot 0.03 + \epsilon_t^{\mu_{bond}}$$

$$r_{t+1}^{rf} = 0.9 \cdot r_t^{rf} + 0.1 \cdot 0.03 + \epsilon_t^{rf}$$

$$\pi_{t+1} = 0.7 \cdot \pi_t + 0.3 \cdot 0.02 + \epsilon_t^\pi$$

All $\epsilon_t^{market} \sim \mathcal{N}(0, \sigma_{market}^2)$ with appropriate volatilities.

**Labor Market Transition**:

$$X_{t+1}^{labor} = 0.9 \cdot X_t^{labor} + 0.1 \cdot 50 + \epsilon_t^{labor}$$

where $\epsilon_t^{labor} \sim \mathcal{N}(0, 10^2)$ (mean-reverting AR(1) around 50).

**Time Transition**: $\tau_{t+1} = \tau_t + 1$

**Competition Schedule**: $F_{t+1} = \max(0, F_t - 1)$ (counts down to competition)

### Reward Function $R_1(s_t, a_t)$

Multi-objective utility function combining 7 components:

$$R_1(s_t, a_t) = \alpha_1 U^{training}(s_t, a_t) + \alpha_2 U^{consumption}(a_t) + \alpha_3 U^{housing}(s_t)$$
$$+ \alpha_4 U^{security}(s_t) + \alpha_5 U^{health}(s_t) + \alpha_6 U^{career}(s_t) - \alpha_7 C^{stress}(s_t)$$

**1. Training Utility** ($\alpha_1 = 8$):
$$U^{training}(s_t, a_t) = 0.4 \cdot \frac{m_t^{snatch} + m_t^{C\&J} + 0.5 \cdot m_t^{squat}}{200} + 0.3 \cdot \frac{\Phi_t}{100} \cdot \frac{v_t^{load}}{500}$$
$$+ 0.2 \cdot (1 - \frac{\Theta_t}{100}) + 0.1 \cdot (a_t^{coaching} + 1) + 50 \cdot a_t^{compete}$$

Combines strength level (40%), training quality (30%), recovery state (20%), coaching quality (10%), and competition participation bonus.

**2. Consumption Utility** ($\alpha_2 = 2$):
$$U^{consumption}(a_t) = 10 \cdot \log(1 + a_t^{dining}/100) + 8 \cdot \log(1 + a_t^{social}/100)$$
$$+ 5 \cdot \log(1 + a_t^{hobbies}/100) + 30 \cdot a_t^{travel}$$

Logarithmic utility from discretionary spending (diminishing returns) and travel bonus.

**3. Housing Utility** ($\alpha_3 = 3$):
$$U^{housing}(s_t) = 0.15 \cdot n_t^{nightlife} + 0.15 \cdot n_t^{cafes} + 0.20 \cdot n_t^{walkability}$$
$$+ 0.10 \cdot n_t^{parks} + 0.15 \cdot h_t^{quality} - 0.10 \cdot h_t^{noise}$$
$$+ 15 \cdot h_t^{laundry} + 10 \cdot h_t^{storage} - 0.05 \cdot (c_t^{work} + c_t^{gym})$$

Values neighborhood amenities, housing quality, and penalizes commute time.

**4. Financial Security Utility** ($\alpha_4 = 2.5$):
$$U^{security}(s_t) = 15 \cdot \sqrt{\frac{W_t}{10000}} + 10 \cdot \sqrt{\frac{E_t}{10000}}$$
$$+ 5 \cdot \sqrt{\frac{p_t^{stocks} + p_t^{bonds}}{10000}} + 3 \cdot \sqrt{\frac{r_t^{401k} + r_t^{IRA}}{10000}} - 20 \cdot \sqrt{\frac{D_t}{1000}}$$

CRRA utility (diminishing returns) from wealth, emergency fund, investments, retirement savings; debt penalty.

**5. Health Utility** ($\alpha_5 = 2$):
$$U^{health}(s_t) = 0.3 \cdot Q_t + 0.25 \cdot \Nu_t + 0.25 \cdot \Phi_t + 0.20 \cdot \Psi_t - 30 \cdot \frac{\Gamma_t}{100}$$

Values sleep, nutrition, physical/mental energy; severe injury penalty.

**6. Career Utility** ($\alpha_6 = 1.5$):
$$U^{career}(s_t) = 0.5 \cdot \frac{S_t - 60000}{140000} + 0.3 \cdot j_t^{performance} + 0.2 \cdot X_t^{labor}$$

Values salary growth, job performance, and favorable labor market conditions.

**7. Stress Cost** ($\alpha_7 = 3$):
$$C^{stress}(s_t) = (\frac{\Sigma_t}{100})^2 + 40 \cdot \mathbb{1}_{W_t < 3000} \cdot \mathbb{1}_{E_t < 6 \cdot h_t^{rent}}$$

Quadratic stress disutility and acute financial insecurity penalty (low liquid wealth + insufficient emergency fund).

**Total Reward Range**: Approximately [-50, 250] per biweek, with training and housing quality dominating for high-performing policies.

### Discount Factor
$\gamma = 0.995$ (biweekly discounting over 3-year horizon: 78 time steps)

---

## RL Version (Phase 3)

### MDP Specification: $\mathcal{M}_2 = (S_2, A_2, P_2, R_2, \gamma)$

**Goal**: Solvable with modern deep RL (PPO, SAC) by end of course, while capturing realistic complexity.

### State Space $S_2$
Mixed discrete/continuous, 21-dimensional state space:

$$s_t = (W_t, E_t, I_t, R_t, L_t, h_t^{rent}, h_t^{commute}, W_t^{intensity}, \Phi_t, \Psi_t, \Theta_t, \Sigma_t, m_t^{snatch}, m_t^{C\&J}, V_t, F_t, S_t, j_t^{perf}, M_t^{market}, Q_t, \tau_t)$$

**Financial State** (4 vars):
- $W_t \in [0, 200000]$: Liquid wealth (continuous, $)
- $E_t \in \{0, 1, 2\}$: Emergency fund level (0=none, 1=partial, 2=full ≥6mo expenses)
- $I_t \in [0, 500000]$: Investment portfolio value (continuous, aggregate stocks/bonds, $)
- $R_t \in [0, 300000]$: Retirement savings (continuous, aggregate 401k/IRA, $)

**Housing/Location State** (3 vars):
- $L_t \in \{0, 1, ..., 7\}$: Neighborhood ID (8 discrete zones: Malvern/Paoli, Wayne/Berwyn, Ardmore, West Chester, King of Prussia, Radnor/Villanova, Devon/Frazer, Conshohocken)
- $h_t^{rent} \in [1000, 3500]$: Monthly rent (continuous, $)
- $h_t^{commute} \in [10, 120]$: Total commute time to work + gym (continuous, minutes)

**Work State** (1 var):
- $W_t^{intensity} \in [0, 100]$: Work intensity/demand this period (stochastic, exogenous)

**Energy/Capacity State** (5 vars):
- $\Phi_t \in [0, 100]$: Physical energy (continuous)
- $\Psi_t \in [0, 100]$: Mental energy (continuous)
- $\Theta_t \in [0, 100]$: Accumulated training fatigue (continuous)
- $\Sigma_t \in [0, 100]$: Stress level (continuous)
- $Q_t \in [0, 100]$: Sleep quality (continuous)

**Training/Performance State** (4 vars):
- $m_t^{snatch} \in [40, 140]$: Snatch 1RM (continuous, kg)
- $m_t^{C\&J} \in [50, 180]$: Clean & Jerk 1RM (continuous, kg)
- $V_t \in [0, 500]$: Aggregate training load index (continuous, tons/week equivalent)
- $F_t \in \{0, 1\}$: Competition this quarter (binary)

**Career State** (2 vars):
- $S_t \in [60000, 200000]$: Annual salary (continuous, $)
- $j_t^{perf} \in [0, 100]$: Job performance score (continuous, simplified from seniority levels)

**Market State** (1 var):
- $M_t^{market} \in \{0, 1, 2\}$: Market regime (0=bear, 1=neutral, 2=bull)

**Time** (1 var):
- $\tau_t \in \{0, 1, ..., 77\}$: Biweek (3-year horizon)

**Total: 21 state dimensions** (mixed discrete/continuous)

### Action Space $A_2$
Mixed discrete/continuous, 9-dimensional action space:

$$a_t = (a_t^{housing}, a_t^{invest}, a_t^{allocation}, a_t^{emergency}, a_t^{intensity}, a_t^{coaching}, a_t^{nutrition}, a_t^{discretionary}, a_t^{compete})$$

**Housing** (1 action, yearly decision):
- $a_t^{housing}$: If moving, choose from 8 suburban zones + rent tier (discrete set, ~24 options)

**Financial** (3 actions, biweekly):
- $a_t^{invest} \in [0, S_t]$: Investment amount (continuous)
- $a_t^{allocation} \in [0, 100]$: % in stocks (continuous, rest in bonds)
- $a_t^{emergency} \in [0, S_t]$: Emergency fund contribution (continuous)

**Training** (3 actions, biweekly):
- $a_t^{intensity} \in \{0, 1, 2\}$: Training intensity (maintenance/moderate/aggressive)
- $a_t^{coaching} \in \{0, 1\}$: Online coaching (binary)
- $a_t^{nutrition} \in \{0, 1, 2\}$: Nutrition budget tier (basic/moderate/premium)

**Consumption** (2 actions, biweekly):
- $a_t^{discretionary} \in [0, S_t]$: Discretionary spending budget (continuous)
- $a_t^{compete} \in \{0, 1\}$: Compete this quarter (binary)

**Total: 9 action dimensions** (mixed discrete/continuous)

### Transition Dynamics $P_2(s_{t+1} | s_t, a_t)$

**Financial Transitions**:
$$W_{t+1} = W_t + \frac{S_t}{26} - \frac{\text{rent}(L_t)}{26} - a_t^{discretionary} - a_t^{invest} - a_t^{emergency} + \epsilon_t^W$$

$$I_{t+1} = I_t + a_t^{invest} + r_{t+1}(M_t^{market}, a_t^{allocation}) \cdot I_t$$

where $r_{t+1} \sim \mathcal{N}(\mu(M_t^{market}, a_t^{allocation}), \sigma^2)$ (stochastic market returns)

**Work Intensity Transition** (stochastic, exogenous):
$$W_{t+1}^{intensity} = 0.7 \cdot W_t^{intensity} + 0.3 \cdot 50 + \epsilon_t^{work}$$

where $\epsilon_t^{work} \sim \mathcal{N}(0, 15^2)$ (mean-reverting AR(1) process around 50)

**Energy Transitions**:
$$\Phi_{t+1} = \text{clip}(\Phi_t - 0.5 \cdot W_t^{intensity}/100 - 0.3 \cdot V_t - 0.01 \cdot h_t^{commute} + 20 \cdot \mathbb{1}_{Q_t > 80} + \epsilon_t^\Phi, 0, 100)$$

$$\Psi_{t+1} = \text{clip}(\Psi_t - 0.6 \cdot W_t^{intensity}/100 - 0.5 \cdot \Sigma_t + 0.2 \cdot Q_t + \epsilon_t^\Psi, 0, 100)$$

$$\Theta_{t+1} = \text{clip}(\Theta_t + V_t - \text{recovery}(a_t^{nutrition}, Q_t), 0, 100)$$

Note: Energy is **depleted by stochastic work intensity** (not chosen), creating periods of high/low energy availability.

**Training Transitions**:
$$m_{t+1}^{snatch} = m_t^{snatch} + \text{adaptation}(V_t, a_t^{coaching}, \Theta_t, \Phi_t, a_t^{nutrition}) + \epsilon_t^M$$

$$V_t = f(a_t^{intensity}, \Phi_t, a_t^{coaching})$$

(training volume depends on chosen intensity, available energy, coaching)

**Career Transitions**:
$$S_{t+1} \sim \begin{cases}
S_t \cdot (1 + 0.15 + \epsilon_t^S) & \text{if promoted} \\
S_t \cdot (1 + 0.03 + \epsilon_t^S) & \text{otherwise (annual raise)}
\end{cases}$$

where $P(\text{promoted}) = \text{logistic}(\beta_0 + \beta_1 j_t^{performance} + \beta_2 \tau_t + \beta_3 j_t^{level})$

Note: Promotion depends on **performance** (function of energy, stress, time at job), not chosen effort.

**Market Transitions**: 3-state Markov chain with persistence

**Health Transitions**:
$$Q_{t+1} = f(\text{noise}(L_t), \Sigma_t, \Theta_t, W_t^{intensity}) + \epsilon_t^Q$$

### Reward Function $R_2(s_t, a_t)$

$$R_2(s_t, a_t) = \alpha_1 U^{training}(m_t^{snatch}, m_t^{C\&J}, V_t, \Phi_t) + \alpha_2 U^{consumption}(a_t^{discretionary})$$
$$+ \alpha_3 U^{housing}(L_t) + \alpha_4 U^{security}(W_t, E_t, I_t) + \alpha_5 U^{health}(\Phi_t, \Psi_t, Q_t) - \alpha_6 \Sigma_t^2$$

Where:
- $U^{training} = 5 \cdot (m_t^{snatch} + m_t^{C\&J})/200 + 3 \cdot (\Phi_t/100) \cdot (V_t/500) + 2 \cdot (1 - \Theta_t/100)$
  (combines strength gains and training quality, high weight $\alpha_1 = 8$)
- $U^{consumption} = 10 \cdot \log(1 + a_t^{discretionary}/100)$
  (log utility from discretionary spending, $\alpha_2 = 2$)
- $U^{housing} = \text{quality\_index}(L_t) \in [0, 100]$
  (neighborhood amenities/nightlife/walkability mapped from discrete zone, $\alpha_3 = 3$)
- $U^{security} = 15 \cdot \sqrt{W_t/10000} + 10 \cdot E_t + 5 \cdot \sqrt{I_t/10000}$
  (CRRA utility from wealth and emergency fund, $\alpha_4 = 2.5$)
- $U^{health} = 0.35 \cdot \Phi_t + 0.35 \cdot \Psi_t + 0.30 \cdot Q_t$
  (physical/mental energy and sleep quality, $\alpha_5 = 2$)
- Stress disutility: $\alpha_6 \Sigma_t^2/10000$ (quadratic, $\alpha_6 = 3$)

### Discount Factor
$\gamma = 0.998$ (biweekly discounting over 3-year horizon: 78 time steps)

---

## DP Version (Phase 2)

### MDP Specification: $\mathcal{M}_3 = (S_3, A_3, P_3, R_3, \gamma)$

**Goal**: Tractable with exact dynamic programming (backward value iteration). To achieve this, three variables from the RL version are dropped: salary is fixed as a known external parameter (starting salary with scheduled annual raises baked into the wealth transition), market regime is replaced by a fixed expected return, and emergency fund is absorbed into the wealth variable. This removes three state dimensions while preserving the core tradeoffs.

### State Space $S_3$
Discrete, 6-dimensional state space:

$$s_t = (W_t, L_t, W_t^{intensity}, \Phi_t, P_t, \tau_t)$$

- $W_t \in \{0, 1, ..., 10\}$: Wealth level (units of $\$20k$, range $\$0$–$\$200k$; wealth above $\$200k$ is clipped to bin 10)
- $L_t \in \{0, 1, 2\}$: Neighborhood tier (0=cheap/far, 1=medium, 2=expensive/close to gym)
- $W_t^{intensity} \in \{0, 1, 2\}$: Work intensity this month (low/medium/high, exogenous stochastic)
- $\Phi_t \in \{0, 1, 2\}$: Physical energy level (low/medium/high)
- $P_t \in \{0, 1, 2\}$: Performance readiness (undertrained/on-track/peaked)
- $\tau_t \in \{0, 1, ..., 35\}$: Month (3-year horizon)

**State space size**: $|S_3| = 11 \times 3 \times 3 \times 3 \times 3 \times 36 = 32{,}076$ states

**Training Cycle phase** is deterministic in $\tau_t$ and does not add a state dimension:
$$\text{phase}(\tau_t) = \tau_t \bmod 3 \in \{0=\text{accumulation},\ 1=\text{intensification},\ 2=\text{peak month}\}$$

Each 3-month block corresponds to one 12-week training cycle, with 12 complete cycles over the 3-year horizon.

### Action Space $A_3$
Fully discrete, 4-dimensional action space (unchanged from V2):

$$a_t = (a_t^{housing}, a_t^{invest}, a_t^{training}, a_t^{consumption})$$

- $a_t^{housing} \in \{-1, 0, 1\}$: Housing decision (downgrade/stay/upgrade neighborhood tier)
- $a_t^{invest} \in \{0, 1, 2\}$: Investment level (0%, 10%, 20% of monthly take-home)
- $a_t^{training} \in \{0, 1, 2\}$: Training intensity (light/moderate/intense)
- $a_t^{consumption} \in \{0, 1, 2\}$: Discretionary spending (frugal/moderate/generous)

**Action space size**: $|A_3| = 3 \times 3 \times 3 \times 3 = 81$ actions per state

### Transition Dynamics $P_3(s_{t+1} | s_t, a_t)$

**Deterministic transitions**:

$$W_{t+1} = \text{clip}\!\left(W_t + \Delta W_t,\ 0,\ 10\right)$$

where $\Delta W_t$ is rounded to the nearest bin and computed as:
$$\Delta W_t = \frac{1}{20000}\Big(S_{base} \cdot r_{raise}^{\lfloor\tau_t/12\rfloor} - \text{rent}(L_t) - \text{spend}(a_t^{consumption}) + \bar{r} \cdot W_t \cdot 20000 - \text{invest}(a_t^{invest})\Big)$$

with $S_{base} = \$140k/12 \approx \$11{,}667$/month, annual raise rate $r_{raise} = 1.03$, fixed expected monthly return $\bar{r} = 0.005$, and:
- $\text{rent}(0,1,2) = (\$1{,}500,\, \$2{,}200,\, \$3{,}000)$/month (Main Line suburbs range)
- $\text{spend}(0,1,2) = (\$1{,}000,\, \$2{,}000,\, \$3{,}500)$/month (frugal/moderate/generous)
- $\text{invest}(0,1,2) = (0,\, 0.10,\, 0.20) \times S_{base}$/month

$$L_{t+1} = \text{clip}(L_t + a_t^{housing},\ 0,\ 2), \qquad \tau_{t+1} = \tau_t + 1$$

**Stochastic transitions** (multinomial tables):

- **Work Intensity** (exogenous — the key stochastic driver):

$$P(W_{t+1}^{intensity} | W_t^{intensity}) = \begin{pmatrix} 0.6 & 0.3 & 0.1 \\ 0.2 & 0.6 & 0.2 \\ 0.1 & 0.3 & 0.6 \end{pmatrix}$$

  (rows = current intensity 0/1/2, columns = next intensity 0/1/2)

- **Energy** $P(\Phi_{t+1} | \Phi_t, W_t^{intensity}, a_t^{training}, L_t)$:
  - High work intensity and high training deplete energy; expensive neighborhood (short commute) provides a bonus
  - Example: $P(\Phi_{t+1}=0 \mid \Phi_t=2,\, W_t^{intensity}=2,\, a_t^{training}=2,\, L_t=0) = 0.7$
  - Example: $P(\Phi_{t+1}=2 \mid \Phi_t=1,\, W_t^{intensity}=0,\, a_t^{training}=1,\, L_t=2) = 0.5$

- **Performance Readiness** $P(P_{t+1} \mid P_t,\, a_t^{training},\, \Phi_t,\, \text{phase}(\tau_t))$:

  The training cycle phase shapes which transitions are possible:
  - **Accumulation** ($\text{phase}=0$): $P$ resets after a competition — begins at 0 or 1 regardless of previous peak. High training and energy can push $P$ from 0 to 1.
  - **Intensification** ($\text{phase}=1$): Consistent training + adequate energy allows $P$ to build from 1 to 2. High work intensity risks dropping $P$ (can't recover between sessions).
  - **Peak month** ($\text{phase}=2$): $P$ is mostly carried forward. Training should taper (moderate intensity preserves $P$; heavy training can cause $P$ to drop).

  Example transition probabilities during intensification ($\text{phase}=1$):

  | $P_t$ | $a_t^{training}=2,\, \Phi_t=2$ | $a_t^{training}=1,\, \Phi_t=1$ | $a_t^{training}=0$ |
  |-------|-------------------------------|-------------------------------|---------------------|
  | 0 | $P(P_{t+1}=1)=0.7$ | $P(P_{t+1}=1)=0.3$ | $P(P_{t+1}=0)=1.0$ |
  | 1 | $P(P_{t+1}=2)=0.5$ | $P(P_{t+1}=1)=0.8$ | $P(P_{t+1}=0)=0.2$ |
  | 2 | $P(P_{t+1}=2)=0.8$ | $P(P_{t+1}=2)=0.6$ | $P(P_{t+1}=1)=0.6$ |

  High $W_t^{intensity}$ shifts all probabilities one column to the left (degraded training quality).

### Reward Function $R_3(s_t, a_t)$

$$R_3(s_t, a_t) = \alpha_1 P_t + \alpha_2 \Phi_t + \alpha_3 L_t + \alpha_4 \sqrt{W_t} - \alpha_5 \cdot \mathbb{1}_{W_t = 0} + \beta \cdot P_t \cdot \mathbb{1}_{\text{phase}(\tau_t)=2}$$

- $\alpha_1 = 3$: Performance readiness (ongoing training quality)
- $\alpha_2 = 2$: Energy (daily wellbeing and training capacity)
- $\alpha_3 = 1.5$: Neighborhood quality (commute and amenities, proxied by location tier)
- $\alpha_4 = 2$: Wealth security (CRRA-style diminishing returns; $W_t$ indexes $\$0$–$\$200k$)
- $\alpha_5 = 8$: Financial distress penalty (zero wealth, i.e., $W_t = 0$)
- $\beta = 10$: **Peak-timing bonus** — large reward for $P_t = 2$ during peak month ($\text{phase}=2$)

The peak-timing bonus is the key innovation: it creates an incentive to build readiness *before* the peak month, forcing the agent to plan training and energy management across the full 3-month cycle. A policy that trains hard at the wrong time or lets energy collapse during busy work periods will fail to peak on schedule.

**Terminal reward**: $R_T(s_T) = 10 \cdot W_T + 5 \cdot P_T$ (final wealth and peak readiness)

The four reward components map directly to the four action dimensions: training intensity drives $P_t$, investment drives $W_t$, housing choice drives $L_t$, and discretionary spending is constrained by $W_t$.

### Discount Factor
$\gamma = 0.99$ (monthly discounting over 3-year horizon: 36 time steps)

### Solution Method
- **Exact**: Backward value iteration with full state enumeration (32,076 states × 81 actions = ~2.6M Q-values, trivially fast)
- **Sensitivity analysis**: Vary $\alpha$ weights and transition probabilities to assess policy robustness

---

## Code Implementation

The MDP is implemented as a Python class `MultiCapitalEnergyMDP` extending the course's `MarkovDecisionProcess[S, A]` base class. See separate file `multi_capital_mdp.py` for the full implementation. Currently implements V3; V2 and V1 are left as future extensions.

**State and Action Spaces:**
- `StateV3`: Fully discrete 6-dimensional frozen dataclass (hashable for DP enumeration)
  - Fields: `wealth` (0–10), `location` (0–2), `work_intensity` (0–2), `energy` (0–2), `performance` (0–2), `time` (0–35)
- `ActionV3`: Fully discrete 4-dimensional frozen dataclass
  - Fields: `housing` (−1/0/1), `invest` (0–2), `training` (0–2), `consumption` (0–2)

**Transition Dynamics:**
- `step(state, action)` returns `Distribution[Tuple[State, float]]`
- Stochastic components (multinomial tables):
  - **Work intensity**: Exogenous 3×3 Markov chain (not chosen by agent)
  - **Energy**: Depends on work intensity, training choice, and location tier
  - **Performance readiness**: Phase-aware 3×3 table (phase = `time % 3`)
- Deterministic components: wealth accumulation (fixed salary + returns − expenses), location choice, time step

**Reward Function:**
- Six-term scalar reward: $\alpha_1 P_t + \alpha_2 \Phi_t + \alpha_3 L_t + \alpha_4 \sqrt{W_t} - \alpha_5 \mathbb{1}_{W_t=0} + \beta P_t \mathbb{1}_{\text{phase}=2}$
- Weights $\alpha_i$ and peak bonus $\beta$ are constructor parameters (default values match proposal)
- Terminal reward: $10 W_T + 5 P_T$

**State Space:**
- $11 \times 3 \times 3 \times 3 \times 3 \times 36 = 32{,}076$ states
- $3 \times 3 \times 3 \times 3 = 81$ actions per state
- ~2.6M Q-values total — tractable with exact backward value iteration

### Example Usage

```python
from multi_capital_mdp import MultiCapitalEnergyMDP, StateV3, NonTerminal

# Create MDP instance
mdp = MultiCapitalEnergyMDP(gamma=0.99, time_horizon=36)

# Create initial state (wealth=$40k → bin 2, medium location, medium work, high energy, on-track, month 0)
state = NonTerminal(StateV3(
    wealth=2, location=1, work_intensity=1,
    energy=2, performance=1, time=0
))

# Get available actions and sample transition
actions = list(mdp.actions(state))  # up to 81 actions
next_dist = mdp.step(state, actions[0])
next_state, reward = next_dist.sample()
```
