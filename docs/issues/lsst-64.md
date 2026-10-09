# Issue Working Document Template


# LSST-64 — Generate a rubin_scheduler cloud database in which every month is average

| Field | Value |
|-------|--------|
| **Issue** | [LSST-64](https://rubinobs.atlassian.net/browse/LSST-64) |
| **Branch** | `tickets/lsst-64` |
| **Author** | Eric H. Neilsen, Jr. |
| **Status** | Implementation |
| **Scope Tier** | T2 |
| **QA Level** | Low |
| **Estimate** | 2 days |
| **Created / Updated** | <2026-10-07> / <2026-10-07> |

---

## 1. Abstract

Current simulations using `rubin_scheduler` use cloud values read from
a database constructed from historical cloud records at CTIO. Some
historical times are much better or worse than others, for any given
year or time period within a year. This is acceptable for simulations
if the goal is to measure metrics after several (or all) years of the
survey, but if the goal is to compare actual progress against a
reference baseline, such a process will not work well as a reference:
deviations from the reference are as likely to come from deviations
from average within the baseline as they are from deviations from
average in weather as experienced. Instead, the code generated in this
issue will use stochastic matrices derived from historical data to
generate a database of simulated cloud values in which every month is
typical.

### 1.1 Scope

**In scope.**

- Code to read a table of historical records and write a database of cloud values suitable for use by `rubin_scheduler` simulations.
- A shell CLI that takes a file of historical records as an argument, and writes the database of cloud levels in the format and schema accepted by `rubin_scheduler`.
- A python API that supports generation of intermediary data, such that the process can be run with calls to a jupyter notebook and intermediary results examined therein.
- Generation of diagnostic plots to verify that the distributions of cloud-levels for each month in the simulated data match the global distributions in the historical data for those months, to the extent possible given the number of quarters in a month (anywhere from 112 to 124).
- Generation of dignostics that verify that the distributions of transitions (cloud level for quarter n-1 to cloud level for quarter n) for any given month in the simulated data approximately match the relative distributions of transitions in historical data.

**Out of scope.**

- Refactoring the handling or use of cloud data in `rubin_scheduler`.
- Collection of cloud data from sources other than the historical CTIO records provided.
- Execution or analysis of scheduler simulations.

**Done when.**

The issue will be complete when a user can generate an cloud database
in which every month is typical of that month with diagnostics that
verify that this is the case, given the table with the historical CTIO
cloud data.

---

## 2. Concept of Operations

There are two operation scenarios:

1. A member of the Rubin Observatory scheduler team run the CLI at the
   Rubin Observatory USDF hosted at SLAC's S3DF, and generates the new
   clouds database. This cloud database will then be used for
   scheduler simulations (actual execution of these simulations is
   outside the scope of this issue).

2. A member of the Rubin Observatory scheduler team runs a jupyter
   notebook running in the "notebook aspect" of the USDF. This
   notebook calls the different stages in the pipeline that implements
   the generation of the database, interactively examines the
   intermediat data products, and writes the resultant database to
   disk.

Although primarily targeted at the USDF, it is expected that this
utility could be run anywhere the `rubin_sim` with it present
dependencies is installed, and where the table of historical cloud
data is readable.

---

## 3. External Requirements and Design Evaluation Criteria

Guidance (T2): at most 8 requirements (process §2.3).

### 3.1 External Requirements

#### R-1: Read historical CTIO cloud-cover data

The command shall read historical cloud-cover data from a space-separated-value text file with column headings, for example starting:

```
sday	eday	month	year	q1	q2	q3	q4
1	2	1	1975	0	0	0	0
2	3	1	1975	0	2	3	4
3	4	1	1975	0	3	4	0
```

The columns shall the following meanings:

| Cloumn name | type | description |
| ----------- | ---- | ----------- |
| sday        | int  | day of the month (in local time) on which the night began |
| eday        | int  | day of the month (in local time) on which the night ended |
| month       | int  | the month of the year on which the night began (1 is January, etc.) |
| year        | int  | the year on which the night began |
| q1          | int  | cloud level in eighths recorded (9 for missing data) for the 1st quarter of the night |
| q2          | int  | cloud level in eighths recorded (9 for missing data) for the 2nd quarter of the night |
| q3          | int  | cloud level in eighths recorded (9 for missing data) for the 3rd quarter of the night |
| q4          | int  | cloud level in eighths recorded (9 for missing data) for the 4th quarter of the night |

The path of the file from which the historical data is to be loaded shall be passed by the user as an argument.
If this file is missing data for all years for any month, the program shall exit with an error.

#### R-2: Write a `rubin_scheduler` compatible cloud database

The command shall write a `rubin_scheduler` compatible cloud database is an SQLite3 database with the following schema:

```
CREATE TABLE Cloud(cloudId INTEGER PRIMARY KEY,c_date INTEGER,cloud DOUBLE, source TEXT);
```

The columns shall have the following meanings:

| Cloumn name | type | description |
| ----------- | ---- | ----------- |
| cloudId     | INTEGER | incrementing numeric index |
| c_date      | INTEGER | seconds past the start of the first year of the simulation|
| cloud       | DOUBLE  | fractional cloud cover, from 0 to 1 |
| source      | TEXT  | Indicates source, in this case always "simulation" |

Note that "cloud" values will always be a multiple of 1.0/8.

To act as a replacement for the current cloud database derived from
historical data, c_date shall be seconds past 1975-01-01T00:00:00 TAI
for the purpose of month boundries, leap years, and the times at
centres of quarters of nights. Night start and end are defined by 12
degree twilight at Cerro Pachon.

For the purposes of repeat alignment, a mismatch of up to one day
between in month boundries is acceptable, so the 365.25 repeat period
implemented in rubin_scheduler is acceptable.

The length of the simulated database shall be specified by a
user-supplied paramater as an integer number of years, `num_years`. If
necessary for the database to cover at least `num_years * 31557600`
seconds (`max(c_date) - min(c_date) < num_years * 31557600`), the
simulation shall add one additional month. This will prevent the
wrapping code in `rubin_scheduler.site_models.cloud_data.CloudData`
from failing.


#### R-3: Distribution of clouds by month

The relative distribution of clouds for every simulated month shall
match the global distribution of clouds for all instances of that
month in the historical database, to within one count (quarter).

Each quarter night shall contribute to the measured distribution of
clouds values on the starting calendar day of that night (sday).
Missing data (specified by a sentinel value of "9" in the historical
record files) shall be excluded.

#### R-4: Transition matrix

For each simulated month, the number of transitions from one cloud
state to the next shall match the the relative frequency of
transitions in the historical record for that month to within the
square root of the expected number of transitions (average historical
frequency of transitions from the specified cloud state to the
specified next scaled to the simulated month longth) in the simulated
month. (A closer match is preferable.) If the program cannot achieve
such a match, it may raise an error.

For example, if 31% of transitions ending in a quarter in a night
starting in October were transitions from 1/8 to 0/8, then a simulated
October should have close to 0.31 * 31 * 4 = 38.44 such transitions, to
within sqrt(38.44)=6.2: anything from 33 to 44 such transitions would be
acceptable. (When the simulation covers many years, a roughly even
mixture of Octobers with 38 and 39 such transitions would be ideal such
that the global proportion of statistics are as close as possible to
the global proportion of statistics in the historical sample, but
explicit limits on the global statistics in the total simulated
database is outside the scope of this requirement except in so far as
they are a consequence of the requirement on individual months.)

If there are no transitions between valid states for a given month in
the historical sequence, the program may exit with an error.

Transitions to or from quarters with missing data (a sentinal value of
"9" in the historical record) shall be excluded from the statistics.
If quarters are separated by one or more missing quarters, they are
not considered a pair.  Transitions from the 4th quarter of one night
to the first quarter of the next is considered a pair.

Any transition pair for which the second quarter of the pair has a
starting night in a given month shall be included in that month's
statistics.

Transition fractions should be calculated by combining all included
pairs for that month for all years, not by grouping individually by
month.

For outgoing transition from states from which there are no pairs in
historical evidence, the transition probability may fall-back to
probabilites from other origins, with the following
fallback priorities:

| starting state  | fallback starting states  |
|:--|:--|
| 0  | 1, 2, 3, 4, 5, 6, 7, 8  |
| 1  | 2, 0, 3, 4, 5, 6, 7, 8  |
| 2  | 1, 0, 3, 4, 5, 6, 7, 8  |
| 3  | 4, 5, 6, 7, 8, 2, 1, 0  |
| 4  | 3, 5, 6, 7, 8, 2, 1, 0  |
| 5  | 6, 7, 4, 8, 3, 2, 1, 0  |
| 6  | 5, 7, 4, 8, 3, 2, 1, 0  |
| 7  | 6, 5, 8, 4, 3, 2, 1, 0  |
| 8  | 7, 6, 5, 4, 3, 2, 1, 0  |


Falling back to a different starting state does not generate an
exception to transition count requirments, or cloud distribution count
requirements stated in R-3.

#### R-5: Performance

The generation of a 20 year simulated database and diagnostic plots
from an input historical data set with 17414 nights
(`clouds/clouds_1975-01-01_to_2022-09-04.txt` in the
`rubin_sim_notebooks` repository) shall take less than one hour
when executed as a slurm batch job with these parameters:

```
#SBATCH --partition=milano              # Partition (queue) names
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --ntasks=1                      # Number of tasks run in parallel
#SBATCH --cpus-per-task=1               # Number of CPUs per task
#SBATCH --mem=12G                       # Requested memory
#SBATCH --time=1:00:00                 # Wall time (hh:mm:ss)
```

#### R-6: Environment

The command shall run in a python `venv` on the USDF S3DF development
node with `rubin_sim` installed with current dependencies (including
optional dependencies).

#### R-7: Intermediary data products

The API must provide access to the following intermediate data:

1. The historical data itself as a `pandas.Series` or a column in a `pandas.DataFrame` with one element for each quarter, indexed by date (year, month, and day of the start of the night) and quarter.
2. The transition count matrix by month for the historical data as a `pandas.DataFrame` indexed by month and starting cloud state.
3. The empirical stochastic matrix by month for the historical data as a `pandas.DataFrame` indexed by month and starting cloud state.
4. The simulated data as a `pandas.DataFrame` indexed by date and quarter, with one column of the same shape as the historical data in item 1, and a `c_date` column with the time as saved in the output database.
5. The transition count matrix by month for the simulated data as a `pandas.DataFrame` indexed by month and starting cloud state.


#### R-8: Diagnostics

The following plots shall be generated for diagnostics, written to
disk when run with the CLI (when paths for the plot files are set with
command line arguments) or creatable with API calls within a jupyter
notebook.

1. An array of historgrams (one for each month) overplotting the normalized distribution of simulated clouds over the distribution of historical clouds for each month.
2. An array of transition count matrix count 2-d histograms, with one pair for each month, showing the transition count matrices for each month for historical and simulated data. The API should permit limiting simulated data to specific years, or combining that month for all years. The CLI should just show the all-year 2-d histograms.

### 3.2 Design Evaluation Criteria
Omit this subsection when only one design is plausible for this issue — that is expected for most T1/T2 issues (process §2.3, §2.4). When this subsection is omitted, the Frame and Design phases are normally performed in a single combined session (process §4, §5.2, §5.3).
Prompts:
- Criteria for comparing alternative designs
- Trade-off principles
- Constraints that shape acceptable solutions

---

## 4. Context

### `rubin_scheduler.site_models.cloud_data`

- Simulated clouds interpret `c_date` in the output database as seconds past <start_year>-01-01, TAI (`rubin_scheduler.site_models.cloud_data.CloudData.__init__`).
- Simulated clouds return the cloud value from the nearest time to the requested one, as measured in seconds past the start of the first year (`rubin_scheduer.site-models.cloud_data.CloudData.__call__` and `rubin_scheduer.site-models.cloud_data.CloudData.read_data`.
- Simulated clouds past the database range reapeat the database at intervals of `floor(db_duration / 31557600) * 31557600` seconds (`rubin_scheduer.site-models.cloud_data.CloudData.read_data`, `rubin_scheduer.site-models.cloud_data.CloudData.__call__`)

### Cloud data notebooks and prototype

- Prior work importing cloud databases can be found on the `tickets/LSST-64` branch of the `rubin_sim_notebooks` repository in the `clouds` directory.
- An incomplete experimental prototype for average cloud generation can be found in `clouds/generate_average_clouds.ipynb` in the `tickets/LSST-64` branch of the `rubin_sim_notebooks` repository. Note that a the transition across months differers from that specified in R-4. R-4 should be followed with this implementation, not the prototype.

---

## 5. Critical Design Decisions (when applicable)
List key decisions where **genuine alternatives** exist. Guidance (T2): at most 3 decisions. Omit this section when no genuine alternative exists; a single plausible option belongs in Architecture and Design (§6) as a design fact, not here (process §2.3, §2.4).
Prompts:
- Decision question
- Options considered
- Final choice and rationale
- Risks and alternatives

---

## 6. Architecture and Design

### Architucture

The program shall follow a pipe and filter architecture: it shall
consist of a sequence of functions called successively, each consuming
either raw input or the output of functions prior to it in the
sequence.

There shall be two supported orchestration modes:

1. There will be a high-level orchestration function that calls each
   function in turn, and a CLI interface implemented with `click` to
   call this orchestration function.
2. A jupyter notebook may call each function through its python API,
   supporting interactive exploration by developers.

### Data Dictionary

#### Type hints

Type hints shall be defined for each schema of `pandas.Series` or
`pandas.DataFrame` using using `typing.TypeAlias`. 

The schema shall be described in comments in the code near the
`typing.TypeAlias` line for each, but the schema need not be enforced
in the code itself, but shall be checked in CI tests.

#### `Clouds`

The historical record read from the input file shall be stored in a `pandas.DataFrame` with the following index levels and columns:

| name    | index? | type | description |
|:--------|:-------|:-----|:------------|
| year    | Y      | np.uint16 | The historical year (local calendar date at the start of the night) |
| month   | Y      | np.uint8  | The month number (1 is January, local calendar date at the start of the night) |
| sday    | Y      | np.uint8  | The day of the month (local calendar date at the start of the night) |
| quarter | Y      | np.uint8  | The quearter of the night (1 through 4) |
| eighths | N      | np.uint8  | the cloud cover in integer 8ths, with 9 as a sentinel for missing data |
| cloud   | N      | np.double | the fractional cloud cover for the quarter (np.nan if eighths=9, eighths/8 otherwise) |
| c_date  | N      | np.uintp  | the number of seconds past 1975-01-01 00:00 TAI of the center of the quarter. |

#### `TransitionMatrix`

A `TransitionMatrix` holds the counts of transitions from one state to another for a series of cloud values.
There shall be two instances:

- `historical_transition_matrix` counts transitions in the historical input data.
- `simulation_transition_matrix` counts transitions in the simululated (output) data.

A `TransitionMatrix` shall have a two level index:

| name  | type | description |
|:------|:-----|:------------|
| month | np.uint8  | The month number (1 is January, local calendar date at the start of the night) |
| origin | np.unit8 | The cloud state (in eighths) for the state at the start of the counted transition |

The name of the column index shall be 'destination', and the column names shall integer cloud cover states (0 through 8)

Transitions to or from the sentinal value for missing data (9) shall
be excluded from the transition matrix entirely: only transitions from
consecutive valid values shall be counted.

All columns will be of `np.uintp` dtype.

#### `StochasticMatrix`

A `StochasticMatrix` is a `pandas.DataFrame` holds the right stochastic matrix with transition
probabities from an origin state (eighths cloud cover) to a
destination state (eighths cloud cover) for each month of the year.

The `StochasticMatrix` shall have a two level index:

| name  | type | description |
|:------|:-----|:------------|
| month | np.uint8  | The month number (1 is January, local calendar date at the start of the night) |
| origin | np.unit8 | The cloud state (in eighths) for the state at the start of the counted transition |

The name of the column index shall be 'destination', and the column names shall integer cloud cover states (0 through 8)

All columns will be of `np.float64` dtype.

Each row must sum to 1.0.

#### `CloudDistribution`

A `CloudDistribution` is a `pandas.DataFrame` indexed by month (as an
integer), with columns for each cloud state (0 through 8, excluding
missing data marked by 9).  The value of each column shall be the
number of instances in that cloud state in that month.

### High-level functions

#### `rubin_sim.clouds.read_historical_clouds`

Accepts a path to one or more files with historical data (in the format defined in R-1), and returns the corresponding instance of `Clouds`.
If multiple paths are given, the data from the files are combined, taking the first instance when there are duplicate values for a given night.

#### `rubin_sim.clouds.count_cloud_states`

Accepts an instance of `Clouds` and returns a corresponding instance of `CloudDistribution`.

#### `rubin_sim.clouds.compute_transition_matrix`

Returnst the transition matrix for a given `Clouds`.

```python
def compute_transition_matrix(
    clouds: Clouds,
	prior_state: int | None,
	**kwargs: Any
	) -> TransitionMatrix
```

`prior_state` defines the cloud state if the quarter immediately
preceeding the provided sequence. If it is either `None` or `9`, the
transition into the first state is excluded from the matrix.


#### `rubin_sim.clouds.compute_stochastic_matrix`

Accepts in instance of `TransitionMatrix` and returns the corresponding empirical stochastic matrix as a `StochasticMatrix`.

#### `rubin_sim.clouds.generate_average_clouds`

Accepts instances of `TransitionMatrix`, `CloudDistribution`, and an integer representing a number of years, and returns an instance of `Clouds` meeting requirments R-3 and R-4.

#### `rubin_sim.clouds.save_clouds`

Accepts a file path and an instance of `Clouds`, and writes an `sqlite3` database meeting requirement R-2.

#### `rubin_sim.clouds.plot_cloud_histogram`

Generates plots filling requirement R-8.1. Follows this call signature:

```python
def plot_cloud_histogram(
    historical_distribution: CloudDistribution,
	simulated_distribution: CloudDistribution | dict[int, CloudDistribution],
	**kwargs: Any
	) -> matplotlib.figure.Figure
```

were the dictionary option of `simulated_distribution` maps years to clouds distributions.

#### `rubin_sim.clouds.plot_transition_histograms`

Accepts two instances of `TransitionMatrix` (one historical, one simulated) ang generates a `matplotlib` figure filling requirement R-8.2.
It should accept additional argument in `**kwargs` and pass them to relevant `matplotlib` plotting functions.

#### `rubin_sim.clouds.cloud_generation_workflow`

An orchestration function that accepts paths to historical data and an
output database and optional arguments for diagnosics, and runs the
full workflow, writing the diagnostics if the optional arguments are
provided.


### Average cloud generation algorithm.

The cloud generation algorithm shall be that of the prototype notebook
in `rubin_sim_notebooks/clouds/generate_average_clouds.ipynb`, except
that:

- there should be a limit to the number of times the while loop executes with a given len(cloud_seq), and throws an exception if that limit is exceeded.
- When the loop exits (fills cloud_seq) it should test whether the requirements for the distribution of cloud state and numbers of transitions are met by it (R-3 and R-4), and throw an exception if they are not.

The initial state of 0 is only an initial seed to get the sequence
started: the transition into the first state should not be included in
the first month transition matrix. This has the result that the number
of transitions in the first month will be one less than the number of
quarters in the month, while for all other months the number of
transitions is the number of quarters.


This generation function should be called in a loop with limited
retries until it succeeds without throwing either of the above
specified exceptions, or reaching the limiting number of
retries. Successive iterations through the loop should retain the same
random number generator and not reset it (which would defeat the
point).


### Requirements verification

| Requirement | Verification                                                                                                                         |
|:------------|:-------------------------------------------------------------------------------------------------------------------------------------|
| R-1         | Unit test for  `rubin_sim.clouds.read_historical_clouds` passes                                                                      |
| R-2         | Unit test for `rubin_sim.clouds.save_clouds` passes                                                                                  |
| R-3         | End-to-end unit tests running `rubin_sim.clouds.cloud_generation_workflow` passes                                                    |
| R-4         | End-to-end unit tests running `rubin_sim.clouds.cloud_generation_workflow` passes                                                    |
| R-5         | Human verification by submission of batch test unit test script that then passes in specified time                                   |
| R-6         | Human verification using the same batch script used in R-5                                                                           |
| R-7         | Unit tests of `read_historical_clouds`, `compute_transition_matrix`, `compute_stochastic_matrix`, and `generate_average_clouds` pass |
| R-8         | Unit tests of `plot_cloud_histogram` and `plot_transition_histograms` pass                                                           |

Most of these verification steps will be tested through standard unit
tests needed in implementation anyway.  The exception is that
implementation also needs to create a bash script with an appropriate
slum header that sets up the environment and runs the
`rubin_sim.clouds.cloud_generation_workflow` unit test.


<!--

Provide the traditional, human-approved implementation contract, properly scaled to the scope and risk of the issue. One page is a target for ordinary T2 issues; additional detail is justified when it improves implementation or human understanding. Describe the delta, not existing behavior already captured in Context.

The design should include, as applicable: design summary; affected components and responsibilities; control and data flow; interfaces and data; behavior and invariants; implementation outline; risks, rollout, and migration; and the implementation decision envelope. Genuine consequential alternatives and their human rationale may be recorded in §5 or here.

**Every External Requirement (§3.1) must be mapped here to a verification method** — the test (unit, integration, or end-to-end) that will demonstrate it, or, where automated testing is impractical, the explicit manual-check steps that will. This mapping is part of the design, not an afterthought for Implementation. For T1, a one-line note per requirement suffices; for T2, use a short table, e.g.:

| Requirement | Verification |
|---|---|
| <requirement id/summary> | <unit test name/description, or manual-check steps> |

Prompts:
- Components, interfaces, algorithms
- Data structures and workflow descriptions
- Preconditions, invariants, and postconditions
- IEEE 1471-2000 style "views"
- UML or traditional software engineering diagrams in mermaid
- Examples or doctests where helpful
- Requirement-to-test mapping (required — see above)

-->

---

## 7. Acceptance Criteria and Evidence
Testable criteria confirming completion. Required whenever an IWD exists (T1 and above — T0 has no IWD, see process §2). Guidance (T2): at most 8 criteria (process §2.3).

Each criterion must be demonstrated by an actual test result, not merely asserted. Record the evidence here or in the requirement-to-verification table in §6; do not duplicate the same criterion merely to preserve template symmetry.

Prompts:
- Required behaviors (covering all "External Requirements"), each with a cited passing test or recorded manual-check result
- Edge cases
- API expectations
- Performance thresholds
- Non-functional criteria

---

## 8. Open Questions
Gaps an agent could not resolve, raised here rather than answered by guessing or resolved only in conversation (process §3.4). Tactical questions that do not affect architecture, external requirements, or acceptance criteria may be resolved by the agent and noted in the Implementation Notes. Write `None.` if there are no outstanding blocking questions. Resolve each with a developer answer, or promote it to a Critical Design Decision (§5) if it turns out to be a genuine trade-off.

**Q-1.** <question>
- *Impact:* <what is blocked until this is answered>
- *Answer:* <developer's answer, or "→ see §5">

---

## 9. Notes, Risks, and Future Considerations (Optional)
Prompts:
- Known risks and ambiguities
- Expected extensions
- Dependencies on other issues
- Assumptions that may change

---

## 10. Change Log (Optional)
Also the record of anything promoted to project documentation during Documentation (process §5.6) — one line per item, naming the target file.
Prompts:
- Date / author / summary of changes
- Harvested items: target file and short description

---

## Definition of Done

- [ ] Scope (§1.1) respected — nothing implemented from the out-of-scope list.
- [ ] Architecture and Design (§6) approved; for T2 before implementation, for T1 before merge.
- [ ] Every External Requirement (§3.1) has a passing test or a recorded manual check (§6, §7).
- [ ] Material deviations from Architecture and Design (§6) are approved and recorded.
- [ ] A targeted diff review was completed; detailed review was performed for applicable risk triggers.
- [ ] Durable content has been promoted to project documentation where useful and logged in the Change Log (§10).
- [ ] CI is green; the change has been reviewed per the review discipline (process §4.1).


