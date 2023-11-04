package main 
 
import ( "testing"
         "github.com/stretchr/testify/assert"
)

func TestFruitDistribution(t *testing.T) {
    assert := assert.New(t)
    assert.Equal(8, FruitDistribution("5 apples and 6 oranges", 19))
    assert.Equal(10, FruitDistribution("5 apples and 6 oranges", 21))
    assert.Equal(2, FruitDistribution("0 apples and 1 oranges", 3))
    assert.Equal(2, FruitDistribution("1 apples and 0 oranges", 3))
    assert.Equal(95, FruitDistribution("2 apples and 3 oranges", 100))
    assert.Equal(0, FruitDistribution("2 apples and 3 oranges", 5))
    assert.Equal(19, FruitDistribution("1 apples and 100 oranges", 120))
}
