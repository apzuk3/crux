package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"

	"github.com/apzuk3/crux"
)

// All forecasts, schedules, prices, and availability below are fictional fixtures.
const tripDate = "2026-09-19"

type cityArgs struct {
	City string `json:"city" description:"Boston or Philadelphia"`
	Date string `json:"date" description:"Trip date in YYYY-MM-DD format"`
}

type Forecast struct {
	Summary         string `json:"summary"`
	TemperatureC    int    `json:"temperature_c"`
	RainProbability int    `json:"rain_probability_pct"`
}

type Transport struct {
	ID                  string `json:"id"`
	City                string `json:"city"`
	Mode                string `json:"mode"`
	ArrivalTime         string `json:"arrival_time"`
	ReturnDepartureTime string `json:"return_departure_time"`
	PricePerPersonCents int    `json:"round_trip_price_per_person_cents"`
}

type Activity struct {
	ID                  string `json:"id"`
	City                string `json:"city"`
	Name                string `json:"name"`
	Indoor              bool   `json:"indoor"`
	StartTime           string `json:"start_time"`
	EndTime             string `json:"end_time"`
	PricePerPersonCents int    `json:"price_per_person_cents"`
}

var transports = []Transport{
	{"bos-train", "Boston", "train", "10:30", "18:00", 12000},
	{"bos-bus", "Boston", "bus", "11:00", "18:30", 8500},
	{"phl-train", "Philadelphia", "train", "09:00", "19:00", 7000},
	{"phl-bus", "Philadelphia", "bus", "09:30", "18:30", 4000},
}

// Activities are ordered by recommendation. The first indoor option in each
// city is sold out, so the model must check availability and try a replacement.
var activities = []Activity{
	{"bos-science", "Boston", "Science museum special exhibit", true, "12:00", "14:00", 3500},
	{"bos-art", "Boston", "Art museum collection tour", true, "12:00", "14:00", 3000},
	{"bos-harbor", "Boston", "Harbor walking tour", false, "15:00", "16:30", 2000},
	{"phl-history", "Philadelphia", "History museum special exhibit", true, "11:00", "13:00", 2500},
	{"phl-art", "Philadelphia", "Art museum collection tour", true, "11:00", "13:00", 3000},
	{"phl-market", "Philadelphia", "Indoor market tasting tour", true, "14:00", "15:30", 2500},
	{"phl-walk", "Philadelphia", "Old City walking tour", false, "16:00", "17:00", 1500},
}

var remainingSpots = map[string]int{
	"bos-science": 0, "bos-art": 8, "bos-harbor": 12,
	"phl-history": 0, "phl-art": 6, "phl-market": 4, "phl-walk": 10,
}

func validateDate(date string) error {
	if date != tripDate {
		return fmt.Errorf("mock data is only available for %s", tripDate)
	}
	return nil
}

func validateCity(args cityArgs) error {
	if err := validateDate(args.Date); err != nil {
		return err
	}
	if args.City != "Boston" && args.City != "Philadelphia" {
		return fmt.Errorf("unsupported city %q; use Boston or Philadelphia", args.City)
	}
	return nil
}

func getWeather(ctx context.Context, args cityArgs) (Forecast, error) {
	log.Printf("get_weather(%+v)", args)
	if err := validateCity(args); err != nil {
		return Forecast{}, err
	}
	if args.City == "Boston" {
		return Forecast{"Heavy rain throughout the afternoon", 16, 90}, nil
	}
	return Forecast{"Afternoon showers likely", 22, 70}, nil
}

func searchTransport(ctx context.Context, args cityArgs) ([]Transport, error) {
	log.Printf("search_transport(%+v)", args)
	if err := validateCity(args); err != nil {
		return nil, err
	}
	options := []Transport{}
	for _, option := range transports {
		if option.City == args.City {
			options = append(options, option)
		}
	}
	return options, nil
}

type activityArgs struct {
	City       string `json:"city" description:"Boston or Philadelphia"`
	Date       string `json:"date" description:"Trip date in YYYY-MM-DD format"`
	IndoorOnly bool   `json:"indoor_only" description:"Only return indoor activities when true"`
}

func searchActivities(ctx context.Context, args activityArgs) ([]Activity, error) {
	log.Printf("search_activities(%+v)", args)
	if err := validateCity(cityArgs{args.City, args.Date}); err != nil {
		return nil, err
	}
	options := []Activity{}
	for _, activity := range activities {
		if activity.City == args.City && (!args.IndoorOnly || activity.Indoor) {
			options = append(options, activity)
		}
	}
	return options, nil
}

type availabilityArgs struct {
	ActivityID string `json:"activity_id" description:"An ID returned by search_activities"`
	Date       string `json:"date" description:"Trip date in YYYY-MM-DD format"`
	PartySize  int    `json:"party_size" description:"Number of adults, greater than zero"`
}

type Availability struct {
	Available      bool `json:"available"`
	RemainingSpots int  `json:"remaining_spots"`
}

func checkAvailability(ctx context.Context, args availabilityArgs) (Availability, error) {
	log.Printf("check_availability(%+v)", args)
	if err := validateDate(args.Date); err != nil {
		return Availability{}, err
	}
	if args.PartySize <= 0 {
		return Availability{}, fmt.Errorf("party_size must be greater than zero")
	}
	spots, ok := remainingSpots[args.ActivityID]
	if !ok {
		return Availability{}, fmt.Errorf("unknown activity ID %q; use search_activities first", args.ActivityID)
	}
	return Availability{spots >= args.PartySize, spots}, nil
}

type costArgs struct {
	TransportID string   `json:"transport_id" description:"An ID returned by search_transport"`
	ActivityIDs []string `json:"activity_ids" description:"Distinct activity IDs in the same city as the transport"`
	Date        string   `json:"date" description:"Trip date in YYYY-MM-DD format"`
	PartySize   int      `json:"party_size" description:"Number of adults, greater than zero"`
}

type TripCost struct {
	Currency       string `json:"currency"`
	TransportCents int    `json:"transport_cents"`
	ActivityCents  int    `json:"activity_cents"`
	TotalCents     int    `json:"total_cents"`
}

func calculateTripCost(ctx context.Context, args costArgs) (TripCost, error) {
	log.Printf("calculate_trip_cost(%+v)", args)
	if err := validateDate(args.Date); err != nil {
		return TripCost{}, err
	}
	if args.PartySize <= 0 || len(args.ActivityIDs) == 0 {
		return TripCost{}, fmt.Errorf("provide a positive party_size and at least one activity")
	}
	var selected Transport
	for _, option := range transports {
		if option.ID == args.TransportID {
			selected = option
			break
		}
	}
	if selected.ID == "" {
		return TripCost{}, fmt.Errorf("unknown transport ID %q; use search_transport first", args.TransportID)
	}
	cost := TripCost{Currency: "USD", TransportCents: selected.PricePerPersonCents * args.PartySize}
	seen := make(map[string]bool)
	for _, id := range args.ActivityIDs {
		if seen[id] {
			return TripCost{}, fmt.Errorf("duplicate activity ID %q", id)
		}
		seen[id] = true
		var selectedActivity Activity
		for _, activity := range activities {
			if activity.ID == id {
				selectedActivity = activity
				break
			}
		}
		if selectedActivity.ID == "" || selectedActivity.City != selected.City {
			return TripCost{}, fmt.Errorf("activity %q must be a known activity in %s", id, selected.City)
		}
		if remainingSpots[id] < args.PartySize {
			return TripCost{}, fmt.Errorf("activity %q has insufficient availability; choose a replacement", id)
		}
		cost.ActivityCents += selectedActivity.PricePerPersonCents * args.PartySize
	}
	cost.TotalCents = cost.TransportCents + cost.ActivityCents
	return cost, nil
}

func init() {
	crux.RegisterTool("get_weather", "Get a fictional forecast for the trip date", getWeather)
	crux.RegisterTool("search_transport", "Find fictional same-day round trips from New York; times are local destination times", searchTransport)
	crux.RegisterTool("search_activities", "Find fictional activities in recommendation order; availability must be checked separately", searchActivities)
	crux.RegisterTool("check_availability", "Check whether an activity has room for the whole party; does not book tickets", checkAvailability)
	crux.RegisterTool("calculate_trip_cost", "Calculate transport and activity costs for the whole party in USD cents; excludes meals and local transfers", calculateTripCost)
}

type Output struct {
	Destination string     `json:"destination"`
	Date        string     `json:"date"`
	PartySize   int        `json:"party_size"`
	Forecast    Forecast   `json:"forecast"`
	Transport   Transport  `json:"transport"`
	Activities  []Activity `json:"activities"`
	Cost        TripCost   `json:"cost"`
	Rationale   string     `json:"rationale"`
	Adjustments []string   `json:"adjustments"`
}

func main() {
	agent := crux.Must(crux.New(
		"day-trip-planner",
		crux.ClaudeHaiku4_5,
		crux.WithInstructions(`You are a day-trip planner working with fictional demo data.
Use tools for all forecasts, schedules, prices, and availability. Never invent IDs.
First fetch weather and transport for BOTH candidate cities, grouping independent
calls in the same turn when possible. Then search activities based on the forecasts;
prefer indoor activities when rain probability is at least 50 percent.
Evaluate activities in recommendation order: check the first recommendation before
moving to alternatives. If it is unavailable, check a replacement and explain the change.
Choose at least two non-overlapping activities that fit between arrival and return
departure, allowing at least 30 minutes between transport and activities and between activities.
Check availability for every selected activity for the whole party. Call calculate_trip_cost
with the final selection; if it exceeds the budget, revise the selection and recalculate.
Only return the structured itinerary after availability and cost are verified.
Explain the destination comparison and note that prices exclude meals and local transfers.
This is a plan only; no tickets have been booked.`),
		crux.WithTools([]string{
			"get_weather", "search_transport", "search_activities",
			"check_availability", "calculate_trip_cost",
		}),
		crux.WithMaxTurns(15),
		crux.WithOutputSchemaFrom[Output](),
	))

	var output Output
	session := crux.MustSession(crux.NewSession(agent))
	err := session.RunInto(context.Background(),
		"Plan a day trip from New York for two adults on "+tripDate+". Compare Boston and Philadelphia, "+
			"keep round-trip transport and activities under $300 total, and choose at least two activities "+
			"suitable for the forecast. If an activity is unavailable, find a replacement.",
		&output,
	)
	if err != nil {
		log.Fatal(err)
	}

	encoder := json.NewEncoder(os.Stdout)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(output); err != nil {
		log.Fatal(err)
	}
}
