import { Button, Center, Checkbox, Container, Flex, Overlay, Slider, Text } from "@mantine/core";
import { useState } from "react";

export type ClockSettings = {
  useClock: boolean;
  minutesMs: number;
  incrementMs: number;
  incrementOnTimeoutMs: number;
};

type Props = {
  onConfirm: (settings: ClockSettings) => void;
};

const ClockSetting = ({ onConfirm }: Props) => {
  const [useClock, setUseClock] = useState(true);
  const [minutes, setMinutes] = useState(10);
  const [incrementSeconds, setIncrementSeconds] = useState(15);
  const [incrementOnTimeout, setIncrementOnTimeout] = useState(10);

  const confirm = () => {
    const settings: ClockSettings = {
      useClock: useClock,
      minutesMs: minutes * 60 * 1000,
      incrementMs: incrementSeconds * 1000,
      incrementOnTimeoutMs: incrementOnTimeout * 1000,
    };

    onConfirm(settings);
  };

  return (
    <Overlay backgroundOpacity={0.9}>
      <Center w="100%" h="100%">
        <Container w="100%" size="xs">
          <Flex direction="column" align="center" justify="center" gap={10} w="100%">
            <Checkbox
              label="Uporabi uro"
              checked={useClock}
              onChange={(event) => setUseClock(event.currentTarget.checked)}
              color="brown"
              size="lg"
              styles={(theme) => ({
                label: {
                  color: theme.white,
                },
              })}
            />

            <Text c="white">Minute na igralca: {useClock ? minutes : "∞"}</Text>
            <Slider
              label={null}
              value={minutes}
              onChange={setMinutes}
              min={1}
              max={60}
              step={1}
              w="100%"
              disabled={!useClock}
              color="brown"
            />

            <Text c="white">Dodatek na potezo: {useClock ? incrementSeconds : "/"} s</Text>
            <Slider
              label={null}
              value={incrementSeconds}
              onChange={setIncrementSeconds}
              min={0}
              max={60}
              step={1}
              w="100%"
              disabled={!useClock}
              color="brown"
            />

            <Text c="white">Dodatek po izteku: {useClock ? incrementOnTimeout : "/"} s</Text>
            <Slider
              label={null}
              value={incrementOnTimeout}
              onChange={setIncrementOnTimeout}
              min={0}
              max={60}
              step={1}
              w="100%"
              disabled={!useClock}
              color="brown"
            />

            <Button onClick={confirm} color="brown" mt={20} w="100%">
              Prični igro
            </Button>
          </Flex>
        </Container>
      </Center>
    </Overlay>
  );
};

export default ClockSetting;
