from cps.simulation.factory_line_config import FactoryLineConfig

MEAN_TIME_BETWEEN_FAULTS = 25.0
AGENT_POLL_INTERVAL = 4
OBSERVATION_MONITOR_INTERVAL = 2

DEFAULT_FACTORY_CONFIG = FactoryLineConfig(
	product="car",
	quantity=1000,
	stations=(
		("Sheet-Metal-Press", 5.0),
		("Body-Welding-Cell", 4.5),
		("Paint-Booth", 4.0),
		("Powertrain-Install", 3.5),
		("Interior-Fitout", 3.0),
		("Wheel-And-Tire-Install", 2.5),
		("Final-Inspection", 2.0),
	),
)
