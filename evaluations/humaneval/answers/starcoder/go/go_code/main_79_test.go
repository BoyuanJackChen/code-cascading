package main 
 
import ( "testing"
         "github.com/stretchr/testify/assert"
)

func TestDecimalToBinary(t *testing.T) {
    assert := assert.New(t)
    assert.Equal("db0db", DecimalToBinary(0))
    assert.Equal("db100000db", DecimalToBinary(32))
    assert.Equal("db1100111db", DecimalToBinary(103))
    assert.Equal("db1111db", DecimalToBinary(15))
}
