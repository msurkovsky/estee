import random
import numpy as np


class SchedulerBase:

    def init(self, simulator):
        self.simulator = simulator

    def schedule(self, new_ready, new_finished):
        return ()


class DoNothingScheduler(SchedulerBase):
    pass


class RandomAssignScheduler(SchedulerBase):

    def init(self, simulator):
        self.scheduled = False

    def schedule(self, new_ready, new_finished):
        if self.scheduled:
            return ()
        self.scheduled = True

        tasks = list(self.simulator.task_graph.tasks)
        random.shuffle(tasks)

        workers = self.simulator.workers

        results = []
        for t in tasks:
            results.append((random.choice(workers), t))
        return results


class AllOnOneScheduler(SchedulerBase):

    def schedule(self, new_ready, new_finished):
        worker = self.simulator.workers[0]
        return [(worker, task) for task in new_ready]


class QueueScheduler(SchedulerBase):

    def __init__(self):
        self.ready = []
        self.queue = None

    def init(self, simulator):
        super().init(simulator)
        self.queue = self.make_queue()

    def make_queue(self, simulator):
        raise NotImplementedError()

    def choose_worker(self, workers, task):
        raise NotImplementedError()

    def schedule(self, new_ready, new_finished):
        self.ready += new_ready
        workers = [w for w in self.simulator.workers if not w.assigned_tasks]
        results = []
        while workers and self.ready:
            for t in self.queue[:]:
                if t in self.ready:
                    self.ready.remove(t)
                    self.queue.remove(t)
                    idx = self.choose_worker(workers, t)
                    results.append((workers.pop(idx), t))
                    break
        return results


class RandomScheduler(QueueScheduler):

    def make_queue(self):
        tasks = self.simulator.task_graph.tasks[:]
        random.shuffle(tasks)
        return tasks

    def choose_worker(self, workers, task):
        return random.randrange(len(workers))


class GreedyTransferQueueScheduler(QueueScheduler):

    def choose_worker(self, workers, task):
        costs = np.zeros(len(workers))

        for i in range(len(workers)):
            w = workers[i]
            for inp in task.inputs:
                if w not in inp.info.assigned_workers:
                    costs[i] += inp.size

        return np.random.choice(np.flatnonzero(costs == costs.min()))


class RandomGtScheduler(GreedyTransferQueueScheduler):

    def make_queue(self):
        tasks = self.simulator.task_graph.tasks[:]
        random.shuffle(tasks)
        return tasks


class BlevelGtScheduler(GreedyTransferQueueScheduler):

    def __init__(self, include_size):
        super().__init__()
        self.include_size = include_size

    def make_queue(self):
        def cost_fn1(t):
            return t.duration

        def cost_fn2(t):
            return t.duration + t.size / bandwidth

        bandwidth = self.simulator.connector.bandwitdth
        assign_b_level(self.simulator.task_graph, cost_fn2 if self.include_size else cost_fn1)
        tasks = self.simulator.task_graph.tasks[:]
        random.shuffle(tasks)  # To randomize keys with the same level
        tasks.sort(key=lambda n: n.s_info, reverse=True)
        return tasks


class SchedulerX(SchedulerBase):

    def init(self, simulator):
        super().init(simulator)
        independencies = compute_independent_tasks(simulator.task_graph)
        tab = []
        costs = []
        bandwidth = self.simulator.connector.bandwitdth
        for task, indeps in independencies.items():
            for t in task.inputs:
                tab.append((t.id, task.id))
                costs.append(-t.size / bandwidth)
            for t in indeps:
                tab.append((t.id, task.id))
                costs.append(t.duration + task.duration)
        self.tab = np.array(tab, dtype=np.int32)
        self.costs = np.array(costs)

        """
        workers = self.simulator.workers
        placement = {}
        for task in simulator.task_graph.tasks:
            placement[task] = workers[0]
        """
        placement = np.zeros(self.simulator.task_graph.task_count)
        print(self.placement_cost(placement))

    def placement_cost(self, placement):
        print(self.tab)
        a = placement[self.tab[:, 0]]
        b = placement[self.tab[:, 1]]
        return (self.costs * (a == b)).sum()


def assign_b_level(task_graph, cost_fn):
    for task in task_graph.tasks:
        task.s_info = cost_fn(task)
    graph_dist_crawl(task_graph.leaf_tasks(), lambda t: t.inputs, cost_fn)


def graph_dist_crawl(initial_tasks, nexts_fn, cost_fn):
    tasks = initial_tasks
    while tasks:
        new_tasks = set()
        for task in tasks:
            dist = task.s_info
            for t in nexts_fn(task):
                new_value = max(t.s_info, dist + cost_fn(t))
                if new_value != t.s_info:
                    t.s_info = new_value
                    new_tasks.add(t)
        tasks = new_tasks


def graph_crawl(initial_tasks, nexts_fn, value_fn):
    values = {}

    def compute(state):
        v = values.get(state)
        if v is not None:
            return v
        v = value_fn(state, [compute(s) for s in nexts_fn(state)])
        values[state] = v
        return v

    for state in initial_tasks:
        compute(state)
    return values


def compute_independent_tasks(task_graph):
    def union(state, values):
        values.append(frozenset((state,)))
        return frozenset.union(*values)

    tasks = frozenset(task_graph.tasks)
    up_deps = graph_crawl(task_graph.leaf_tasks(), lambda t: t.inputs, union)
    down_deps = graph_crawl(task_graph.source_tasks(), lambda t: t.consumers, union)
    print(down_deps)
    return {task: tasks.difference(up_deps[task] | down_deps[task])
            for task in task_graph.tasks}
