SELECT Target.targetId, cellId FROM TargetTraffic JOIN Target ON TargetTraffic.targetId = Target.targetId ORDER BY Target.targetId DESC LIMIT 500 OFFSET 30;
SELECT cellId, COUNT(genderId) FROM TargetTraffic JOIN Target ON TargetTraffic.targetId = Target.targetId GROUP BY cellId;