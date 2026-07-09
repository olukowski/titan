use titan_core::{
    ComponentRegistry, EntityId, FixedStepContext, Result, Schedule, SystemWorld, Transform,
    Velocity, World, phase1_component_registry, velocity_integration_system,
};
use titan_math::Vec3;

fn seeded_motion_system(
    world: &mut SystemWorld<'_>,
    commands: &mut titan_core::CommandBuffer,
    ctx: FixedStepContext,
) -> Result<()> {
    let mut rng = ctx.rng;
    for (_, (transform, velocity)) in world.query_mut::<(&mut Transform, &Velocity)>() {
        transform.translation.y += rng.next_f32() * velocity.linear.x;
    }
    if ctx.frame == 1 {
        let spawned = EntityId::from_raw(1000);
        commands.spawn_with_id(spawned);
        commands.insert(
            spawned,
            Transform::from_translation(Vec3::new(rng.next_f32(), 0.0, 0.0)),
        );
    }
    Ok(())
}

fn run(seed: u64) -> (String, String) {
    let registry: ComponentRegistry = phase1_component_registry().unwrap();
    let mut world = World::new(registry);
    for (id, x) in [(EntityId::from_raw(10), 1.0), (EntityId::from_raw(20), 2.0)] {
        let entity = world.spawn_with_id(id).unwrap();
        world
            .insert(entity, Transform::from_translation(Vec3::new(x, 0.0, 0.0)))
            .unwrap();
        world
            .insert(entity, Velocity::new(Vec3::new(0.25, 0.0, 0.0)))
            .unwrap();
    }
    world.set_runtime_metadata(0, seed);

    let mut schedule = Schedule::new();
    schedule.add_system(
        "titan.core.velocity_integration",
        velocity_integration_system,
    );
    schedule.add_system("tests.seeded_motion", seeded_motion_system);
    for frame in 1..=60 {
        schedule
            .run_fixed_step(&mut world, FixedStepContext::new(1.0 / 60.0, frame, seed))
            .unwrap();
    }

    let state = serde_json::to_string(&world.dump_state().unwrap()).unwrap();
    let events = world.event_log().to_jsonl().unwrap();
    (state, events)
}

#[test]
fn same_seed_produces_byte_identical_state_and_structural_event_dumps() {
    let first = run(0x5eed);
    let second = run(0x5eed);

    assert_eq!(first.0, second.0);
    assert_eq!(first.1, second.1);
}

#[test]
fn different_seeds_produce_different_state_dumps_in_test_local_seeded_system() {
    let first = run(1);
    let second = run(2);

    assert_ne!(first.0, second.0);
}
